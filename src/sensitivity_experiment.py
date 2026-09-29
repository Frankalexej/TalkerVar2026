"""Run preregistered raw diagnostics and post-quantization CE dose responses."""
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import uuid
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from src.config import project_path
from src.experiment import write_json
from src.training import seed_everything
from src.vq_data import prepare_vq_data
from src.vq_experiment import run_jobs
from src.raw_clusterability import fit_raw,evaluate_raw,raw_seed_summary
from src.sensitivity_training import train_sensitivity
from src.sensitivity_metrics import evaluate_sensitivity


def trial_name(kp,ks,weight,seed):
    return f'K{kp}_{ks}_lambda_{weight:g}_seed_{seed}'


def run_sensitivity(config,resume=None):
    data = prepare_vq_data(config)
    reference = project_path(config.reference_run)
    old = json.loads((reference/'data.json').read_text())
    assert data.metadata == old, 'Dataset/preprocessing differ from the previous VQ experiment.'
    manifest = data.frame[['source_row','speaker_id','label']].copy()
    for part,ix in data.indices.items():
        manifest.loc[ix,'split'] = part
    pd.testing.assert_frame_equal(manifest,pd.read_csv(reference/'split_manifest.csv'))
    names = ['sensitivity_config.py','sensitivity_models.py','sensitivity_training.py','sensitivity_metrics.py',
             'raw_clusterability.py','sensitivity_experiment.py','vq_data.py','vq_training.py','vq_models.py',
             'vq_metrics.py','vq_experiment.py','models.py','training.py','config.py','evaluation.py','vq_config.py','experiment.py']
    sources = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}
    plan = {'config':asdict(config),'source_hashes':sources,'csv_sha256':data.metadata['csv_sha256'],
            'selection':'None: all capacities and lambdas reported at fixed final epoch',
            'unseen_speaker_normalization':'Not performed; no training statistics for those identities',
            'primary_vq_accuracy':'training-fitted many-to-one', 'raw_cluster_count':5}
    run = project_path(str(resume)) if resume else project_path(config.output_dir)/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if resume:
        assert json.dumps(json.loads((run/'plan.json').read_text()),sort_keys=True)==json.dumps(plan,sort_keys=True),'Resume source/config mismatch'
        if (run/'complete.json').exists():
            return run
    else:
        run.mkdir(parents=True)
        write_json(run/'plan.json',plan)
        write_json(run/'data.json',data.metadata)
        manifest.to_csv(run/'split_manifest.csv',index=False)
        write_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,
                   'versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','scipy','scikit-learn','matplotlib')}})
        (run/'source_snapshot').mkdir()
        for name in names:
            shutil.copy2(Path(__file__).parent/name,run/'source_snapshot'/name)
        joblib.dump({'imputer':data.imputer,'scaler':data.scaler,'features':config.data.features},run/'preprocessing.joblib')
        joblib.dump({'x':{p:data.x[p] for p in ('train','val')},'speaker':{p:data.y[p][:,1] for p in ('train','val')},
                    'n_phonemes':len(data.phonemes)},run/'vq_train_val.joblib')
        joblib.dump({'x':{p:data.x[p] for p in ('train','val')},'y':{p:data.y[p] for p in ('train','val')}},run/'raw_train_val.joblib')
    print(f'RUN DIRECTORY: {run}',flush=True)
    jobs = []
    for method in ('kmeans','gmm'):
        for view in ('pooled','individual','speaker_normalized'):
            for seed in config.seeds:
                name = f'{method}_{view}_seed_{seed}'
                jobs.append((name,fit_raw,(config,run/'raw_train_val.joblib',run/'raw'/name,method,view,seed)))
    raw_fits = run_jobs(jobs,config.workers,'Raw fits')
    write_json(run/'raw_fits.json',raw_fits)
    raw_rows = []
    with threadpool_limits(config.threads):
        for method in ('kmeans','gmm'):
            for view in ('pooled','individual','speaker_normalized'):
                for seed in config.seeds:
                    raw_rows.extend(evaluate_raw(run/'raw'/f'{method}_{view}_seed_{seed}',data,method,view,seed))
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(run/'raw_metrics.csv',index=False)
    raw_seed_summary(raw).to_csv(run/'raw_seed_metrics.csv',index=False)
    # Raw results never change the prespecified VQ jobs or hyperparameters.
    jobs = []
    for kp,ks in config.codebook_pairs:
        for seed in config.seeds:
            for weight in config.lambdas:
                name = trial_name(kp,ks,weight,seed)
                jobs.append((name,train_sensitivity,(config,run/'vq_train_val.joblib',run/'vq'/name,seed,kp,ks,weight)))
    trained = run_jobs(jobs,config.workers,'VQ training')
    write_json(run/'trained.json',trained)
    fit_frame = pd.DataFrame(trained)
    for _,group in fit_frame.groupby(['k_phi','k_s','seed']):
        for key in ('initial_state_sha256','minibatch_order_sha256'):
            assert group[key].nunique() == 1, f'Unmatched dose pair: {key}'
    for _,group in fit_frame.groupby('seed'):
        assert group.initial_neural_sha256.nunique() == 1, 'Neural initialization differs across capacity'
        assert group.minibatch_order_sha256.nunique() == 1
    rows = []
    seed_everything(2026,config.threads)
    with threadpool_limits(config.threads):
        for kp,ks in config.codebook_pairs:
            for weight in config.lambdas:
                for seed in config.seeds:
                    name = trial_name(kp,ks,weight,seed)
                    rows.extend(evaluate_sensitivity(run/'vq'/name,config,data))
                    print(f'Evaluated {name}',flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(run/'vq_metrics.csv',index=False)
    metrics = ['phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi','s_s_nmi','s_phi_nmi','mse']
    pairs = frame[frame.lambda_s > 0].merge(frame[frame.lambda_s == 0],on=['k_phi','k_s','seed','split'],suffixes=('_dose','_zero'))
    deltas = pairs[['k_phi','k_s','seed','split','lambda_s_dose']].rename(columns={'lambda_s_dose':'lambda_s'}).copy()
    for metric in metrics:
        deltas[metric] = pairs[metric+'_dose']-pairs[metric+'_zero']
    deltas.to_csv(run/'paired_dose_effects.csv',index=False)
    large = frame[(frame.k_phi==64)&(frame.k_s==64)]
    small = frame[(frame.k_phi==5)&(frame.k_s==10)]
    paired = large.merge(small,on=['lambda_s','seed','split'],suffixes=('_large','_small'))
    deltas = paired[['lambda_s','seed','split']].copy()
    for metric in metrics:
        deltas[metric] = paired[metric+'_large']-paired[metric+'_small']
    deltas.to_csv(run/'paired_capacity_effects.csv',index=False)
    write_json(run/'complete.json',{'vq_runs':len(trained),'raw_model_fits':sum(len(r['fits']) for r in raw_fits),
                                  'matched_initialization_verified':True})
    return run
