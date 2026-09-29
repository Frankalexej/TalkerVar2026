"""Frozen selection, matched A/B trials, matched baselines, then test evaluation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from src.vq_training import train_vq
from src.vq_metrics import evaluate_vq
from src.vq_baselines import train_baseline, evaluate_baseline


def run_jobs(jobs, workers, label):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): name for name, fn, args in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append({'trial': futures[future], **result})
            print(f'{label}: {len(results)}/{len(jobs)} completed {futures[future]}', flush=True)
    return results


def run_vq_experiment(config, resume=None):
    data = prepare_vq_data(config)
    sources = ['vq_config.py', 'vq_data.py', 'vq_models.py', 'vq_training.py', 'vq_metrics.py',
               'vq_baselines.py', 'vq_experiment.py', 'models.py', 'training.py', 'evaluation.py', 'config.py']
    hashes = {s: hashlib.sha256((Path(__file__).parent/s).read_bytes()).hexdigest() for s in sources}
    plan = {'config': asdict(config), 'source_hashes': hashes, 'csv_sha256': data.metadata['csv_sha256'],
            'selection': 'mean validation MSE + 0.1*(1-speaker code NMI) + 0.05*(sum unused fractions)',
            'phoneme_labels_for_vq': 'evaluation after configuration selection and training only; category count allowed'}
    run_dir = project_path(str(resume)) if resume else project_path(config.output_dir)/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if resume:
        if json.dumps(json.loads((run_dir/'plan.json').read_text()), sort_keys=True) != json.dumps(plan, sort_keys=True):
            raise ValueError('Resume plan/source/data mismatch')
        if (run_dir/'complete.json').exists():
            return run_dir
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir/'plan.json', plan)
        write_json(run_dir/'data.json', data.metadata)
        write_json(run_dir/'environment.json', {'python': sys.version, 'executable': sys.executable,
                   'versions': {p: importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','torch','matplotlib')}})
        (run_dir/'source_snapshot').mkdir()
        for source in sources:
            shutil.copy2(Path(__file__).parent/source, run_dir/'source_snapshot'/source)
        manifest = data.frame[['source_row','speaker_id','label']].copy()
        for part, ix in data.indices.items():
            manifest.loc[ix, 'split'] = part
        manifest.to_csv(run_dir/'split_manifest.csv', index=False)
        joblib.dump({'imputer': data.imputer, 'scaler': data.scaler, 'features': config.data.features}, run_dir/'preprocessing.joblib')
        joblib.dump({'x': {p:data.x[p] for p in ('train','val')},
                    'speaker': {p:data.y[p][:,1] for p in ('train','val')}, 'n_phonemes': len(data.phonemes)}, run_dir/'vq_train_val.joblib')
        joblib.dump({'x': {p:data.x[p] for p in ('train','val')}, 'y': {p:data.y[p] for p in ('train','val')},
                    'phonemes': data.phonemes, 'speakers': data.speakers}, run_dir/'baseline_train_val.joblib')
    print(f'RUN DIRECTORY: {run_dir}', flush=True)
    print(f'Splits: {data.metadata["split_counts"]}; held out: {data.metadata["heldout_speakers"]}', flush=True)
    jobs = []
    for weight in config.lambdas:
        for seed in config.selection_seeds:
            name = f'lambda_{weight:g}_seed_{seed}'
            jobs.append((name, train_vq, (config, run_dir/'vq_train_val.joblib', run_dir/'selection'/name, seed, weight, config.selection_epochs)))
    selection = pd.DataFrame(run_jobs(jobs, config.workers, 'Selection'))
    selection.to_csv(run_dir/'selection_trials.csv', index=False)
    ranking = selection.groupby('lambda_s').selection_score.mean().sort_values()
    weight = float(ranking.index[0])
    # This record is frozen before any phoneme mappings or test metrics.
    write_json(run_dir/'selection.json', {'lambda_s': weight, 'ranking': {str(k):v for k,v in ranking.items()},
               'phoneme_labels_used': False, 'test_results_used': False})
    print(f'Frozen lambda_s={weight}', flush=True)
    jobs = []
    for condition, value in (('A', weight), ('B', 0.0)):
        for seed in config.seeds:
            name = f'{condition}_seed_{seed}'
            jobs.append((name, train_vq, (config, run_dir/'vq_train_val.joblib', run_dir/'final'/name, seed, value, config.epochs)))
    for method in ('gmvae', 'gmvae_shared_w', 'classifier', 'knn'):
        for seed in (config.baseline_seeds if method != 'knn' else config.baseline_seeds[:1]):
            name = f'{method}_seed_{seed}'
            jobs.append((name, train_baseline, (config, run_dir/'baseline_train_val.joblib', run_dir/'baselines'/name, seed, method)))
    results = run_jobs(jobs, config.workers, 'Final training')
    write_json(run_dir/'training_completed.json', results)
    for seed in config.seeds:
        pair = [json.loads((run_dir/'final'/f'{c}_seed_{seed}'/'validation.json').read_text()) for c in ('A','B')]
        for key in ('initial_state_sha256', 'minibatch_order_sha256'):
            assert pair[0][key] == pair[1][key], f'Matched-pair mismatch: {key}'
    rows = []
    seed_everything(2026, config.threads)
    with threadpool_limits(config.threads):
        for condition in ('A','B'):
            for seed in config.seeds:
                rows.extend(evaluate_vq(run_dir/'final'/f'{condition}_seed_{seed}', config, data, condition, seed))
                print(f'Evaluated {condition} seed {seed}', flush=True)
        for method in ('gmvae', 'gmvae_shared_w', 'classifier', 'knn'):
            for seed in (config.baseline_seeds if method != 'knn' else config.baseline_seeds[:1]):
                rows.extend(evaluate_baseline(config, data, run_dir/'baselines'/f'{method}_seed_{seed}', method, seed))
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir/'metrics.csv', index=False)
    frame.groupby(['method','split']).agg({c:['mean','std'] for c in frame.columns if c not in ('method','split','seed')}).to_csv(run_dir/'summary.csv')
    paired = frame[frame.method == 'A'].merge(frame[frame.method == 'B'], on=['seed','split'], suffixes=('_A','_B'))
    metrics = ['phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi','s_phi_nmi','s_s_nmi','mse']
    deltas = paired[['seed','split']].copy()
    for metric in metrics:
        deltas[metric] = paired[metric+'_A'] - paired[metric+'_B']
    deltas.to_csv(run_dir/'paired_differences.csv', index=False)
    write_json(run_dir/'complete.json', {'lambda_s':weight, 'A_B_seeds':list(config.seeds), 'pair_hashes_verified':True})
    return run_dir
