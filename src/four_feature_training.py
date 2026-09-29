"""Separate workers see only training/validation data, never either test set."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from src.config import ExperimentConfig
from src.models import VowelClassifier
from src.four_feature_models import FourFeatureVQVAE, EfficientTwoFactorVaDE
from src.training import seed_everything, fit_neural
from src.vq_training import initialize, state_digest, perplexity
from src.experiment import write_json


def train_four(config,cache,directory,seed,method,k=0,weight=0):
    directory = Path(directory)
    if (directory/'done.json').exists():
        return json.loads((directory/'done.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    data = SimpleNamespace(**joblib.load(cache))
    assert set(data.x)==set(data.y)=={'train','val'}
    assert data.x['train'].shape[1] == 4
    started = time.perf_counter()
    result = {'method':method,'k_phi':k,'k_s':k,'lambda_s':weight if method=='vq' else None,'seed':seed}
    with (directory/'training.log').open('w') as stream, redirect_stdout(stream),redirect_stderr(stream),threadpool_limits(config.threads):
        seed_everything(seed,config.threads)
        if method=='vq':
            model = FourFeatureVQVAE(config,k,k)
            pretrain = initialize(model,data.x['train'],config,seed)
            pd.DataFrame(pretrain).to_csv(directory/'pretraining.csv',index=False)
            result['initial_state_sha256'] = state_digest(model)
            result['initial_neural_sha256'] = hashlib.sha256(b''.join(v.numpy().tobytes() for key,v in model.state_dict().items() if not key.startswith('vq_'))).hexdigest()
            torch.save(model.state_dict(),directory/'initial.pt')
            optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
            xs = {p:torch.from_numpy(v) for p,v in data.x.items()}
            ys = {p:torch.from_numpy(v[:,1]) for p,v in data.y.items()}
            generator = torch.Generator().manual_seed(seed+2000)
            order_hash = hashlib.sha256()
            records = []
            for epoch in range(1,config.epochs+1):
                row = {'epoch':epoch}
                for part in ('train','val'):
                    training = part=='train'
                    model.train(training)
                    order = torch.randperm(len(xs[part]),generator=generator) if training else torch.arange(len(xs[part]))
                    if training:
                        order_hash.update(order.numpy().tobytes())
                    totals,codes = {},{'phi':[],'s':[]}
                    with torch.set_grad_enabled(training):
                        for ix in order.split(config.batch_size):
                            loss,components,out = model.loss(xs[part][ix],ys[part][ix],weight)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Non-finite VQ loss')
                            if training:
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                                optimizer.step()
                            for key,value in components.items():
                                totals[key] = totals.get(key,0)+value.item()*len(ix)
                            for b in codes:
                                codes[b].extend(out['k_'+b].detach().numpy().tolist())
                    row.update({part+'_'+key:value/len(order) for key,value in totals.items()})
                    row.update({part+'_perplexity_'+b:perplexity(codes[b],k) for b in codes})
                records.append(row)
                pd.DataFrame(records).to_csv(directory/'history.csv',index=False)
                if epoch==1 or epoch%10==0:
                    print(f'epoch {epoch}: val MSE={row["val_reconstruction"]:.5f}',flush=True)
            result.update(checkpoint_epoch=config.epochs,checkpoint_rule='fixed final epoch',minibatch_order_sha256=order_hash.hexdigest())
        elif method=='knn':
            model = KNeighborsClassifier(n_neighbors=15,n_jobs=config.threads).fit(data.x['train'],data.y['train'])
            joblib.dump(model,directory/'model.joblib')
        else:
            cfg = ExperimentConfig(data=config.data,model=config.gmvae_model,
                                   training=replace(config.baseline_training,seed=seed,num_threads=config.threads,
                                                    device=config.gmvae_device if method=='gmvae' else config.baseline_training.device))
            write_json(directory/'training_config.json',asdict(cfg))
            model = (EfficientTwoFactorVaDE(4,k,k,cfg.model) if method=='gmvae' else VowelClassifier(4,5,10,cfg.model))
            # The unsupervised worker's labels are dummy zeros, not phoneme or speaker identities.
            model,details = fit_neural(model,method,data,cfg,directory)
            result.update(details,checkpoint_rule='minimum validation original objective')
        if method!='knn':
            result['parameter_count'] = sum(p.numel() for p in model.parameters())
            torch.save({'state_dict':model.state_dict(),'training':result},directory/'model.pt')
        result['elapsed_seconds'] = time.perf_counter()-started
        write_json(directory/'done.json',result)
    return result


def load_four(config,directory):
    directory = Path(directory)
    info = json.loads((directory/'done.json').read_text())
    method,k = info['method'],info['k_phi']
    if method=='knn':
        return joblib.load(directory/'model.joblib'),info
    model = (FourFeatureVQVAE(config,k,k) if method=='vq' else
             EfficientTwoFactorVaDE(4,k,k,config.gmvae_model) if method=='gmvae' else
             VowelClassifier(4,5,10,config.gmvae_model))
    model.load_state_dict(torch.load(directory/'model.pt',weights_only=True,map_location='cpu')['state_dict'])
    return model.eval(),info
