"""Immutable plan, manifest-backed data, train-only workers, then evaluation."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
import uuid
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.config import project_path
from src.experiment import write_json
from src.four_feature_data import prepare_four_feature_data
from src.four_feature_training import train_four
from src.four_feature_metrics import evaluate_four
from src.vq_experiment import run_jobs
from src.training import seed_everything
from src.raw_clusterability import fit_raw, evaluate_raw, raw_seed_summary


def run_four(config,resume=None):
    data = prepare_four_feature_data(config)
    root = Path(__file__).parent
    names = ['four_feature_config.py','four_feature_data.py','four_feature_models.py','four_feature_training.py',
             'four_feature_metrics.py','four_feature_experiment.py','models.py','training.py','config.py','data.py',
             'vq_training.py','vq_models.py','vq_metrics.py','vq_config.py','vq_experiment.py','evaluation.py',
             'sensitivity_models.py','sensitivity_config.py','sensitivity_metrics.py','raw_clusterability.py','experiment.py']
    plan = {'config':asdict(config),'source_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
            'csv_sha256':data.metadata['csv_sha256'],'manifest_sha256':data.metadata['manifest_sha256'],
            'gmvae':'original TwoFactorVaDE; equivalent matrix quadratic forms; CUDA execution',
            'evaluation':'all prespecified settings; training-fitted many-to-one; no test-based selection',
            'ordering':'saved manifest source-row order within each split, shared by all new conditions'}
    run = project_path(str(resume)) if resume else project_path(config.output_dir)/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if resume:
        assert json.dumps(plan,sort_keys=True)==json.dumps(json.loads((run/'plan.json').read_text()),sort_keys=True),'Resume mismatch'
        if (run/'complete.json').exists():
            return run
    else:
        run.mkdir(parents=True)
        write_json(run/'plan.json',plan)
        write_json(run/'data.json',data.metadata)
        shutil.copy2(project_path(config.reference_run)/'split_manifest.csv',run/'split_manifest.csv')
        (run/'source_snapshot').mkdir()
        for name in names:
            shutil.copy2(root/name,run/'source_snapshot'/name)
        write_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,
                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','scipy','scikit-learn')}})
        joblib.dump({'imputer':data.imputer,'scaler':data.scaler,'features':config.data.features},run/'preprocessing.joblib')
        base = {'x':{p:data.x[p] for p in ('train','val')},'y':{p:data.y[p] for p in ('train','val')}}
        joblib.dump(base,run/'supervised_train_val.joblib')
        joblib.dump({**base,'y':{p:np.column_stack([np.zeros(len(v),dtype='int64'),v[:,1]]) for p,v in base['y'].items()}},run/'vq_train_val.joblib')
        joblib.dump({**base,'y':{p:np.zeros_like(v) for p,v in base['y'].items()}},run/'unsupervised_train_val.joblib')
    print(f'RUN DIRECTORY: {run}',flush=True)
    jobs = []
    for k,_ in config.codebook_pairs:
        for seed in config.seeds:
            name = f'gmvae_K{k}_seed_{seed}'
            jobs.append((name,train_four,(config,run/'unsupervised_train_val.joblib',run/'trials'/name,seed,'gmvae',k,0)))
            for weight in config.lambdas:
                name = f'vq_K{k}_lambda_{weight:g}_seed_{seed}'
                jobs.append((name,train_four,(config,run/'vq_train_val.joblib',run/'trials'/name,seed,'vq',k,weight)))
    for seed in config.seeds:
        name = f'classifier_seed_{seed}'
        jobs.append((name,train_four,(config,run/'supervised_train_val.joblib',run/'trials'/name,seed,'classifier')))
    jobs.append(('knn',train_four,(config,run/'supervised_train_val.joblib',run/'trials'/'knn',config.seeds[0],'knn')))
    trained = run_jobs(jobs,config.workers,'Training')
    write_json(run/'trained.json',trained)
    vq = pd.DataFrame([r for r in trained if r['method']=='vq'])
    assert vq.groupby(['k_phi','seed']).initial_state_sha256.nunique().eq(1).all()
    assert vq.groupby('seed').initial_neural_sha256.nunique().eq(1).all()
    assert vq.groupby('seed').minibatch_order_sha256.nunique().eq(1).all()
    raw_jobs = []
    for method in ('kmeans','gmm'):
        for seed in config.seeds:
            name = f'{method}_pooled_seed_{seed}'
            raw_jobs.append((name,fit_raw,(config,run/'supervised_train_val.joblib',run/'raw'/name,method,'pooled',seed)))
    write_json(run/'raw_fits.json',run_jobs(raw_jobs,config.workers,'Raw clustering'))
    seed_everything(2026,config.threads)
    rows,raw = [],[]
    with threadpool_limits(config.threads):
        for r in trained:
            rows.extend(evaluate_four(config,data,run/'trials'/r['trial']))
            print('Evaluated '+r['trial'],flush=True)
        for method in ('kmeans','gmm'):
            for seed in config.seeds:
                raw.extend(evaluate_raw(run/'raw'/f'{method}_pooled_seed_{seed}',data,method,'pooled',seed))
    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
    raw = pd.DataFrame(raw)
    raw.to_csv(run/'raw_metrics.csv',index=False)
    raw_seed_summary(raw).to_csv(run/'raw_seed_metrics.csv',index=False)
    write_json(run/'complete.json',{'neural_runs':len(trained)-1,'knn_fits':1,'raw_fits':len(raw_jobs),'matched_vq_initialization':True})
    return run
