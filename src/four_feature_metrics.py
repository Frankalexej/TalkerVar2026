"""Frozen training-majority mappings and both-factor evaluation on both tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_mutual_info_score
from src.four_feature_training import load_four
from src.sensitivity_metrics import fit_many_to_one
from src.vq_metrics import code_metrics, contingency
from src.vq_training import predict_vq, perplexity
from src.experiment import write_json


@torch.no_grad()
def predict_four(model,method,x,seed):
    if method=='vq':
        return predict_vq(model,x)
    if method=='knn':
        p = model.predict(x)
        return {'k_phi':p[:,0],'k_s':p[:,1]}
    out = {}
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed+200)
    # Maintain the prior evaluation batch size/32 MC samples. Save marginals
    # rather than all 4096 joint probabilities for every token.
    for batch in torch.from_numpy(x).split(512):
        batch = batch.to(device)
        if method=='gmvae':
            q = model.category_probabilities(batch,generator=generator)
            p,s = q.sum(2),q.sum(1)
            mean = model.encode(batch)[0]
            values = {'k_phi':p.argmax(1),'k_s':s.argmax(1),'prob_phi':p,'prob_s':s,
                      'reconstruction':model.decoder(mean),'latent_mean':mean}
        else:
            p,s = model(batch)
            values = {'k_phi':p.argmax(1),'k_s':s.argmax(1)}
        for key,value in values.items():
            out.setdefault(key,[]).append(value.cpu().numpy())
    return {key:np.concatenate(values) for key,values in out.items()}


def evaluate_four(config,data,directory):
    directory = Path(directory)
    if (directory/'metrics.json').exists():
        return json.loads((directory/'metrics.json').read_text())
    model,info = load_four(config,directory)
    method,k = info['method'],info['k_phi']
    if method=='gmvae':
        model.to(config.gmvae_device)
    outputs = {p:predict_four(model,method,x,info['seed']) for p,x in data.x.items()}
    discrete = method in ('vq','gmvae')
    maps,empty = {},{}
    for i,b in enumerate(('phi','s')):
        if discrete:
            maps[b],empty[b] = fit_many_to_one(outputs['train']['k_'+b],data.y['train'][:,i],k,(5,10)[i])
        else:
            maps[b],empty[b] = np.arange((5,10)[i]),np.array([],dtype=int)
    write_json(directory/'mappings.json',{'fit_split':'train','mapping':{b:m.tolist() for b,m in maps.items()},
                                        'unused_train_codes':{b:e.tolist() for b,e in empty.items()}})
    rows = []
    for part in ('val','test_seen','test_unseen'):
        out = outputs[part]
        yp,ys = data.y[part].T
        row = {key:info[key] for key in ('method','seed','k_phi','k_s','lambda_s')}
        row.update(split=part,phoneme_accuracy=float(np.mean(maps['phi'][out['k_phi']]==yp)))
        if part!='test_unseen':
            row['speaker_code_accuracy' if discrete else 'speaker_classifier_accuracy'] = float(np.mean(maps['s'][out['k_s']]==ys))
            if method=='vq' and info['lambda_s']>0:
                row['speaker_classifier_accuracy'] = float(np.mean(out['speaker_logits'].argmax(1)==ys))
        if discrete:
            for b in ('phi','s'):
                codes = out['k_'+b]
                for factor,truth in (('phi',yp),('s',ys)):
                    row.update({f'{b}_{factor}_{key}':v for key,v in code_metrics(codes,truth,k).items()})
                    row[f'{b}_{factor}_ami'] = float(adjusted_mutual_info_score(truth,codes))
                    matrix = contingency(codes,truth,k,np.unique(truth))
                    pd.DataFrame(matrix,columns=np.unique(truth)).to_csv(directory/f'{part}_{b}_by_{factor}.csv',index_label='code')
                counts = np.bincount(codes,minlength=k)
                row[b+'_active_codes'] = int((counts>0).sum())
                row[b+'_perplexity'] = perplexity(codes,k)
                row[b+'_unmapped_fraction'] = float(np.isin(codes,empty[b]).mean())
                write_json(directory/f'{part}_{b}_usage.json',{'counts':counts.tolist(),'unused_codes':np.flatnonzero(counts==0).tolist()})
            error = np.square(out['reconstruction']-data.x[part])
            row['mse'] = float(error.mean())
            row.update({'mse_'+f:float(v) for f,v in zip(config.data.features,error.mean(0))})
        np.savez_compressed(directory/f'{part}_predictions.npz',**out,true_phoneme=yp,true_speaker=ys,
                            source_row=data.frame.iloc[data.indices[part]].source_row.to_numpy())
        rows.append(row)
    write_json(directory/'metrics.json',rows)
    return rows
