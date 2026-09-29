"""Fixed-budget final-epoch VQ training; no phoneme labels or test inputs."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.experiment import write_json
from src.training import seed_everything
from src.vq_training import initialize, state_digest, perplexity
from src.sensitivity_models import QuantizedSpeakerVQVAE


def train_sensitivity(config, cache, directory, seed, k_phi, k_s, weight):
    directory = Path(directory)
    if (directory/'done.json').exists():
        return json.loads((directory/'done.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    data = joblib.load(cache)
    assert set(data) == {'x','speaker','n_phonemes'} and data['n_phonemes'] == 5
    assert set(data['x']) == set(data['speaker']) == {'train','val'}
    started = time.perf_counter()
    with (directory/'training.log').open('w') as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(config.threads):
        seed_everything(seed,config.threads)
        model = QuantizedSpeakerVQVAE(config,k_phi,k_s)
        pretrain = initialize(model,data['x']['train'],config,seed)
        pd.DataFrame(pretrain).to_csv(directory/'pretraining.csv',index=False)
        init_hash = state_digest(model)
        neural_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for k,v in model.state_dict().items() if not k.startswith('vq_'))).hexdigest()
        torch.save(model.state_dict(),directory/'initial.pt')
        optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        xs = {p:torch.from_numpy(v) for p,v in data['x'].items()}
        ys = {p:torch.from_numpy(v) for p,v in data['speaker'].items()}
        generator = torch.Generator().manual_seed(seed+2000)
        order_hash = hashlib.sha256()
        records = []
        for epoch in range(1,config.epochs+1):
            row = {'epoch':epoch}
            for part in ('train','val'):
                training = part == 'train'
                model.train(training)
                order = torch.randperm(len(xs[part]),generator=generator) if training else torch.arange(len(xs[part]))
                if training:
                    order_hash.update(order.numpy().tobytes())
                totals, codes = {}, {'phi':[],'s':[]}
                with torch.set_grad_enabled(training):
                    for ix in order.split(config.batch_size):
                        loss,components,out = model.loss(xs[part][ix],ys[part][ix],weight)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite objective')
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                            optimizer.step()
                        for key,value in components.items():
                            totals[key] = totals.get(key,0.0)+value.item()*len(ix)
                        for branch in codes:
                            codes[branch].extend(out['k_'+branch].detach().numpy().tolist())
                row.update({part+'_'+k:v/len(order) for k,v in totals.items()})
                for branch,size in (('phi',k_phi),('s',k_s)):
                    row[f'{part}_perplexity_{branch}'] = perplexity(codes[branch],size)
            records.append(row)
            pd.DataFrame(records).to_csv(directory/'history.csv',index=False)
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch {epoch}: val MSE={row['val_reconstruction']:.5f}; CE={row['val_speaker_ce']:.5f}",flush=True)
        result = {'seed':seed,'k_phi':k_phi,'k_s':k_s,'lambda_s':weight,
                  'checkpoint_epoch':config.epochs,'checkpoint_rule':'fixed final epoch',
                  'initial_state_sha256':init_hash,'initial_neural_sha256':neural_hash,
                  'minibatch_order_sha256':order_hash.hexdigest(),
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'elapsed_seconds':time.perf_counter()-started}
        torch.save({'state_dict':model.state_dict(),'config':asdict(config),'training':result},directory/'final.pt')
        write_json(directory/'done.json',result)
    return result


def load_sensitivity(directory,config):
    saved = torch.load(Path(directory)/'final.pt',weights_only=True,map_location='cpu')
    info = saved['training']
    model = QuantizedSpeakerVQVAE(config,info['k_phi'],info['k_s'])
    model.load_state_dict(saved['state_dict'])
    return model.eval(),info
