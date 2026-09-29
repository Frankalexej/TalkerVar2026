"""Frozen many-to-one mappings for overcomplete codes, plus invariant metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_mutual_info_score
from src.evaluation import fit_alignment
from src.experiment import write_json
from src.vq_metrics import contingency, code_metrics
from src.vq_training import predict_vq, perplexity
from src.sensitivity_training import load_sensitivity


def fit_many_to_one(codes,labels,size,nclasses):
    counts = contingency(codes,labels,size,np.arange(nclasses))
    mapping = counts.argmax(1)
    empty = counts.sum(1) == 0
    mapping[empty] = np.bincount(labels,minlength=nclasses).argmax()
    return mapping, np.flatnonzero(empty)


def evaluate_sensitivity(directory,config,data):
    directory = Path(directory)
    if (directory/'metrics.json').exists():
        import json
        return json.loads((directory/'metrics.json').read_text())
    model,info = load_sensitivity(directory,config)
    outputs = {part:predict_vq(model,x) for part,x in data.x.items()}
    sizes = {'phi':info['k_phi'],'s':info['k_s']}
    mappings, empty = {}, {}
    for i,b in enumerate(('phi','s')):
        mappings[b],empty[b] = fit_many_to_one(outputs['train']['k_'+b],data.y['train'][:,i],sizes[b],(5,10)[i])
    hungarian = {b:fit_alignment(data.y['train'][:,i],outputs['train']['k_'+b],sizes[b])
                 for i,b in enumerate(('phi','s')) if sizes[b] == (5,10)[i]}
    write_json(directory/'mappings.json',{'fit_split':'train','many_to_one':{b:m.tolist() for b,m in mappings.items()},
               'empty_train_codes':{b:v.tolist() for b,v in empty.items()},'hungarian':{b:m.tolist() for b,m in hungarian.items()}})
    means = {b:outputs['train']['z_'+b].mean(0) for b in ('phi','s')}
    rows = []
    for part in ('val','test_seen','test_unseen'):
        out = outputs[part]
        yp,ys = data.y[part].T
        row = {k:info[k] for k in ('seed','k_phi','k_s','lambda_s')}
        row.update(split=part,phoneme_accuracy=float(np.mean(mappings['phi'][out['k_phi']] == yp)))
        if 'phi' in hungarian:
            row['phoneme_hungarian_accuracy'] = float(np.mean(hungarian['phi'][out['k_phi']] == yp))
        if part != 'test_unseen':
            row['speaker_code_accuracy'] = float(np.mean(mappings['s'][out['k_s']] == ys))
            if info['lambda_s'] > 0:
                row['speaker_classifier_accuracy'] = float(np.mean(out['speaker_logits'].argmax(1) == ys))
            if 's' in hungarian:
                row['speaker_hungarian_accuracy'] = float(np.mean(hungarian['s'][out['k_s']] == ys))
        for branch,size in sizes.items():
            codes = out['k_'+branch]
            for factor,labels in (('phi',yp),('s',ys)):
                row.update({f'{branch}_{factor}_{key}':value for key,value in code_metrics(codes,labels,size).items()})
                row[f'{branch}_{factor}_ami'] = float(adjusted_mutual_info_score(labels,codes))
                classes = np.unique(labels)
                matrix = contingency(codes,labels,size,classes)
                pd.DataFrame(matrix,columns=classes).to_csv(directory/f'{part}_{branch}_by_{factor}.csv',index_label='code')
            counts = np.bincount(codes,minlength=size)
            row[branch+'_active_codes'] = int((counts>0).sum())
            row[branch+'_perplexity'] = perplexity(codes,size)
            row[branch+'_unmapped_fraction'] = float(np.isin(codes,empty[branch]).mean())
            write_json(directory/f'{part}_{branch}_usage.json',{'counts':counts.tolist(),'unused_codes':np.flatnonzero(counts==0).tolist()})
        error = np.square(out['reconstruction']-data.x[part])
        row['mse'] = float(error.mean())
        row.update({'mse_'+f:float(v) for f,v in zip(config.data.features,error.mean(0))})
        if part.startswith('test'):
            for branch in sizes:
                z = {b:torch.from_numpy(out['z_'+b]) for b in sizes}
                z[branch] = torch.from_numpy(np.broadcast_to(means[branch],z[branch].shape).copy())
                with torch.no_grad():
                    reconstructed = torch.cat([model.decoder(torch.cat([p,s],1)) for p,s in zip(z['phi'].split(2048),z['s'].split(2048))]).numpy()
                row['ablate_'+branch+'_mse_delta'] = float(np.square(reconstructed-data.x[part]).mean())-row['mse']
            np.savez_compressed(directory/f'{part}_representations.npz',**out,true_phoneme=yp,true_speaker=ys,
                                source_row=data.frame.iloc[data.indices[part]].source_row.to_numpy())
        rows.append(row)
    write_json(directory/'metrics.json',rows)
    return rows
