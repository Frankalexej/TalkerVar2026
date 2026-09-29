"""K=5 acoustic clustering: pooled, individual and train-speaker normalized."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits
from src.evaluation import fit_alignment
from src.experiment import write_json
from src.training import seed_everything
from src.vq_metrics import code_metrics, contingency
from src.vq_training import perplexity
from src.sensitivity_metrics import fit_many_to_one


def fit_speaker_normalizer(x,speakers):
    stats = {}
    for s in np.unique(speakers):
        values = x[speakers == s]
        scale = values.std(0)
        scale[scale < 1e-8] = 1.0
        stats[int(s)] = {'mean':values.mean(0),'scale':scale}
    return stats


def speaker_normalize(x,speakers,stats):
    result = np.empty_like(x)
    for s in np.unique(speakers):
        if int(s) not in stats:
            raise ValueError('No training normalization statistics for this speaker; refuse test adaptation.')
        mask = speakers == s
        result[mask] = (x[mask]-stats[int(s)]['mean'])/stats[int(s)]['scale']
    return result


def fit_raw(config,cache,directory,method,view,seed):
    directory = Path(directory)
    if (directory/'done.json').exists():
        return json.loads((directory/'done.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    seed_everything(seed,config.threads)
    data = joblib.load(cache)
    assert set(data['x']) == set(data['y']) == {'train','val'}
    x = data['x']['train'].astype(float)
    y = data['y']['train']
    stats = fit_speaker_normalizer(x,y[:,1]) if view == 'speaker_normalized' else None
    if stats is not None:
        x = speaker_normalize(x,y[:,1],stats)
        joblib.dump(stats,directory/'normalization.joblib')
    keys = range(10) if view == 'individual' else [-1]
    models, mappings, diagnostics = {}, {}, []
    with threadpool_limits(config.threads):
        for key in keys:
            mask = y[:,1] == key if key >= 0 else np.ones(len(y),dtype=bool)
            values,labels = x[mask],y[mask,0]
            model = (KMeans(n_clusters=5,n_init=config.kmeans_n_init,random_state=seed)
                     if method == 'kmeans' else GaussianMixture(n_components=5,covariance_type='full',
                         n_init=config.gmm_n_init,max_iter=config.gmm_max_iter,reg_covar=config.gmm_reg_covar,
                         random_state=seed,init_params='kmeans'))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(values)
            codes = model.predict(values)
            many,_ = fit_many_to_one(codes,labels,5,5)
            mappings[key] = {'hungarian':fit_alignment(labels,codes,5),'many':many}
            models[key] = model
            diagnostics.append({'speaker':key,'n_train':len(values),'n_iter':int(model.n_iter_),
                                'converged':bool(getattr(model,'converged_',True)),
                                'train_objective':float(model.inertia_ if method=='kmeans' else model.lower_bound_),
                                'warnings':[str(w.message) for w in caught]})
    joblib.dump({'models':models,'mappings':mappings},directory/'models.joblib')
    write_json(directory/'mappings.json',{str(k):{name:v.tolist() for name,v in item.items()} for k,item in mappings.items()})
    result = {'method':method,'view':view,'seed':seed,'fits':diagnostics}
    write_json(directory/'done.json',result)
    return result


def evaluate_raw(directory,data,method,view,seed):
    directory = Path(directory)
    if (directory/'metrics.json').exists():
        return json.loads((directory/'metrics.json').read_text())
    saved = joblib.load(directory/'models.joblib')
    stats = joblib.load(directory/'normalization.joblib') if view == 'speaker_normalized' else None
    rows = []
    for part in (('val','test_seen','test_unseen') if view == 'pooled' else ('val','test_seen')):
        x,y = data.x[part].astype(float),data.y[part]
        if stats is not None:
            x = speaker_normalize(x,y[:,1],stats)
        predictions = []
        for key,model in saved['models'].items():
            mask = y[:,1] == key if key >= 0 else np.ones(len(y),dtype=bool)
            labels,speakers = y[mask].T
            codes = model.predict(x[mask])
            mapping = saved['mappings'][key]
            row = {'method':method,'view':view,'seed':seed,'split':part,
                   'speaker':data.speakers[key] if key >= 0 else 'all','n':len(codes),
                   'phoneme_hungarian_accuracy':float(np.mean(mapping['hungarian'][codes] == labels)),
                   'phoneme_accuracy':float(np.mean(mapping['many'][codes] == labels)),
                   'phi_phi_ami':float(adjusted_mutual_info_score(labels,codes)),
                   'active_codes':len(np.unique(codes)),'perplexity':perplexity(codes,5)}
            row.update({'phi_phi_'+k:v for k,v in code_metrics(codes,labels,5).items()})
            if key < 0:
                row.update({'phi_s_'+k:v for k,v in code_metrics(codes,speakers,5).items()})
            rows.append(row)
            pd.DataFrame(contingency(codes,labels,5,np.arange(5)),columns=data.phonemes).to_csv(
                directory/f'{part}_{key}_contingency.csv',index_label='cluster')
            predictions.append(pd.DataFrame({'source_row':data.frame.iloc[data.indices[part][mask]].source_row.to_numpy(),
                                             'speaker':speakers,'phoneme':labels,'local_cluster':codes,
                                             'hungarian_phoneme':mapping['hungarian'][codes],'many_phoneme':mapping['many'][codes]}))
        pd.concat(predictions).to_csv(directory/f'{part}_predictions.csv',index=False)
    write_json(directory/'metrics.json',rows)
    return rows


def raw_seed_summary(frame):
    metrics = ['phoneme_hungarian_accuracy','phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_phi_ami','phi_phi_purity','active_codes','perplexity']
    pooled = frame[frame.view != 'individual'].copy()
    rows = []
    for (method,seed,part),g in frame[frame.view == 'individual'].groupby(['method','seed','split']):
        for name,weights in (('individual_micro',g.n),('individual_macro',np.ones(len(g)))):
            row = {'method':method,'view':name,'seed':seed,'split':part,'speaker':'aggregate','n':int(g.n.sum())}
            row.update({m:float(np.average(g[m],weights=weights)) for m in metrics})
            rows.append(row)
    # NMI/ARI above are averages of within-speaker scores, NOT pooled arbitrary local codes.
    return pd.concat([pooled,pd.DataFrame(rows)],ignore_index=True)
