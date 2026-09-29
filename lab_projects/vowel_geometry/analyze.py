"""All-speaker acoustic baseline with four explicitly defined analysis views."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import zlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,
                             homogeneity_completeness_v_measure)
from threadpoolctl import threadpool_limits
from lab_projects.vowel_geometry.metrics import geometry,query_silhouettes


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def analyze_group(run,cfg,view,group,target,indices):
    run = Path(run)
    name = f'{view}__{group}'
    directory = run/'groups'/name
    if (directory/'done.json').exists():
        return json.loads((directory/'done.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    os.environ['LOKY_MAX_CPU_COUNT'] = str(cfg['threads'])
    x = np.load(run/'standardized.npy',mmap_mode='r')[indices]
    y = np.load(run/f'{target}.npy')[indices]
    source = np.load(run/'source_rows.npy')[indices]
    n,k = len(x),len(np.unique(y))
    salt = zlib.crc32(name.encode())
    rng = np.random.default_rng(cfg['fit_sample_seed']+salt)
    fit_ix = np.sort(rng.choice(n,min(n,cfg['fit_sample_max']),replace=False))
    np.save(directory/'fit_source_rows.npy',source[fit_ix])
    rows = []
    with threadpool_limits(cfg['threads']):
        truth = geometry(x,y)
        for seed in cfg['seeds']:
            model = KMeans(n_clusters=k,n_init=cfg['kmeans_n_init'],max_iter=cfg['kmeans_max_iter'],
                           tol=cfg['kmeans_tol'],random_state=seed,algorithm='lloyd').fit(x[fit_ix])
            codes = model.predict(x)
            joblib.dump(model,directory/f'kmeans_seed_{seed}.joblib')
            rng = np.random.default_rng(seed+100000+salt)
            nq = cfg['query_large'] if n>cfg['large_group_threshold'] else cfg['query_small']
            query = np.sort(rng.choice(n,min(n,nq),replace=False))
            shuffled = np.random.default_rng(seed+200000+salt).permutation(y)
            scores = query_silhouettes(x,{'truth':y,'kmeans':codes,'shuffled':shuffled},query,cfg['distance_chunk'])
            homo,comp,vm = homogeneity_completeness_v_measure(y,codes)
            row = {'view':view,'group':str(group),'target':target,'seed':seed,'n':n,'k':k,'n_fit':len(fit_ix),'n_query':len(query),
                   'fit_iterations':int(model.n_iter_),'fit_hit_max_iter':bool(model.n_iter_>=cfg['kmeans_max_iter']),
                   **{'truth_'+key:v for key,v in truth.items()},
                   **{'kmeans_'+key:v for key,v in geometry(x,codes).items()},
                   'ari':float(adjusted_rand_score(y,codes)),'nmi':float(normalized_mutual_info_score(y,codes)),
                   'ami':float(adjusted_mutual_info_score(y,codes)),'homogeneity':float(homo),'completeness':float(comp),'v_measure':float(vm)}
            for key,values in scores.items():
                row[key+'_silhouette'] = float(values.mean())
                row[key+'_negative_fraction'] = float(np.mean(values<0))
            row['silhouette_above_shuffle'] = row['truth_silhouette']-row['shuffled_silhouette']
            matrix = pd.crosstab(pd.Series(codes,name='cluster'),pd.Series(y,name=target))
            row['purity'] = float(matrix.max(axis=1).sum()/n)
            matrix.to_csv(directory/f'contingency_seed_{seed}.csv')
            pd.DataFrame({'source_row':source,'truth':y,'cluster':codes}).to_csv(directory/f'assignments_seed_{seed}.csv',index=False)
            pd.DataFrame({'source_row':source[query],'truth':y[query],'cluster':codes[query],
                          **{key+'_silhouette':v for key,v in scores.items()}}).to_csv(directory/f'silhouette_queries_seed_{seed}.csv',index=False)
            rows.append(row)
    write_json(directory/'done.json',rows)
    return rows


def run_analysis(cfg,resume=None):
    path = ROOT/cfg['csv_path']
    files = ['analyze.py','metrics.py','config.py']
    plan = {'config':cfg,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_hashes':{f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in files},
            'scope':'all available speakers; descriptive full-dataset analysis, no predictive holdout',
            'normalization':'global median imputation then global population-SD standardization; no per-group rescaling'}
    run = ROOT/resume if resume else ROOT/cfg['output_dir']/datetime.now().strftime('%Y%m%d_%H%M%S')
    if resume:
        assert json.loads((run/'plan.json').read_text())==plan,'Plan mismatch'
        if (run/'complete.json').exists():
            return run
    else:
        run.mkdir(parents=True)
        write_json(run/'plan.json',plan)
        (run/'source_snapshot').mkdir()
        for f in files:
            shutil.copy2(Path(__file__).parent/f,run/'source_snapshot'/f)
    full = pd.read_csv(path,dtype={'speaker_id':str,'label':str})
    if full[['speaker_id','label']].isna().any().any():
        raise ValueError('Missing target identities; define an exclusion policy before proceeding')
    features = cfg['features']
    raw = full[features].to_numpy(float)
    missing = (~np.isfinite(raw))|(raw<=0)
    raw[missing] = np.nan
    if np.isnan(raw).all(0).any():
        raise ValueError('Entire feature missing')
    imputer,scaler = SimpleImputer(strategy='median'),StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(raw))
    speakers = sorted(full.speaker_id.unique()); phonemes = sorted(full.label.unique())
    np.save(run/'standardized.npy',x)
    np.save(run/'source_rows.npy',np.arange(len(full)))
    np.save(run/'speaker.npy',pd.Categorical(full.speaker_id,categories=speakers).codes.astype('int64'))
    np.save(run/'phoneme.npy',pd.Categorical(full.label,categories=phonemes).codes.astype('int64'))
    joblib.dump({'imputer':imputer,'scaler':scaler,'features':features},run/'preprocessing.joblib')
    counts = pd.crosstab(full.speaker_id,full.label)
    counts.to_csv(run/'speaker_phoneme_counts.csv')
    missing_frame = pd.DataFrame(missing,columns=features)
    for field in ('speaker_id','label'):
        missing_frame.groupby(full[field]).mean().to_csv(run/f'missing_fraction_by_{field}.csv')
    quality = {'n_tokens':len(full),'n_speakers':len(speakers),'n_phonemes':len(phonemes),'speaker_ids':speakers,'phonemes':phonemes,
               'phoneme_counts':full.label.value_counts().sort_index().to_dict(),'speaker_count_min':int(counts.sum(1).min()),
               'speaker_count_max':int(counts.sum(1).max()),'missing_counts':dict(zip(features,missing.sum(0).tolist())),
               'all_features_missing_rows':int(missing.all(1).sum()),'absent_speaker_phoneme_cells':int((counts==0).sum().sum()),
               'duplicate_token_identifiers':int(full.duplicated(['speaker_id','sentence_id','segment_id']).sum()),
               'imputation_medians':imputer.statistics_.tolist(),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist()}
    write_json(run/'data_quality.json',quality)
    write_json(run/'environment.json',{'executable':sys.executable,'python':sys.version,
               'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']}})
    jobs = [('phoneme_pooled','all','phoneme',np.arange(len(full))),('speaker_pooled','all','speaker',np.arange(len(full)))]
    jobs += [('phoneme_given_speaker',s,'phoneme',np.flatnonzero(full.speaker_id.eq(s))) for s in speakers]
    jobs += [('speaker_given_phoneme',p,'speaker',np.flatnonzero(full.label.eq(p))) for p in phonemes]
    print('RUN DIRECTORY:',run,flush=True)
    print(f'{len(jobs)} groups; {len(cfg["seeds"])} seeds per group',flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        futures = {pool.submit(analyze_group,run,cfg,*job):(job[0],job[1]) for job in jobs}
        for i,future in enumerate(as_completed(futures),1):
            rows.extend(future.result())
            print(f'{i}/{len(jobs)} groups complete: {futures[future]}',flush=True)
    pd.DataFrame(rows).sort_values(['view','group','seed']).to_csv(run/'metrics.csv',index=False)
    write_json(run/'complete.json',{'groups':len(jobs),'clustering_fits':len(rows),'n_tokens':len(full)})
    return run


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config',default='lab_projects/vowel_geometry/config.py'); p.add_argument('--resume')
    args = p.parse_args()
    print('COMPLETE:',run_analysis(runpy.run_path(args.config)['CONFIG'],args.resume),flush=True)
