"""Matched-seed contrasts, historical references, plots and reproducible report."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import project_path
from src.sensitivity_report import table
from src.vq_report import md_table

METRICS = ['phoneme_accuracy','phi_phi_nmi','phi_phi_ami','phi_phi_ari','phi_s_nmi','s_phi_nmi','s_s_nmi','mse']


def paired(a,b,keys,name,metrics=METRICS):
    joined = a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    out = joined[keys].copy()
    out['comparison'] = name
    for metric in metrics:
        out[metric] = joined[metric+'_a']-joined[metric+'_b']
    return out


def make_four_report(run):
    run = Path(run)
    assert (run/'complete.json').exists()
    cfg = json.loads((run/'plan.json').read_text())['config']
    data = json.loads((run/'data.json').read_text())
    metrics = pd.read_csv(run/'metrics.csv')
    tests = metrics[metrics.split.ne('val')].copy()
    disc = tests[tests.method.isin(['vq','gmvae'])].copy()
    disc['lambda_s'] = disc.lambda_s.fillna(-1)
    contrasts = []
    for k in (16,64):
        v = disc[(disc.method=='vq')&(disc.k_phi==k)]
        g = disc[(disc.method=='gmvae')&(disc.k_phi==k)]
        contrasts.append(paired(v[v.lambda_s==1],v[v.lambda_s==0],['seed','split'],f'VQ K{k}: lambda1 - lambda0'))
        contrasts.append(paired(v[v.lambda_s==0],g,['seed','split'],f'K{k}: unsupervised VQ - GMVAE'))
    for method,weight in (('vq',0),('vq',1),('gmvae',-1)):
        d = disc[(disc.method==method)&(disc.lambda_s==weight)]
        contrasts.append(paired(d[d.k_phi==64],d[d.k_phi==16],['seed','split'],f'{method} lambda{weight}: K64 - K16'))
    effects = pd.concat(contrasts,ignore_index=True)
    effects.to_csv(run/'paired_effects.csv',index=False)
    old = pd.read_csv(project_path(cfg['reference_run'])/'vq_metrics.csv')
    old = old[(old.k_phi==64)&old.lambda_s.isin([0,1])&old.split.ne('val')]
    now = tests[(tests.method=='vq')&(tests.k_phi==64)]
    historical = paired(now,old,['seed','split','lambda_s'],'VQ64: four features - five features',METRICS[:-1])
    historical.to_csv(run/'historical_vq_effects.csv',index=False)
    references = tests[tests.method.isin(['classifier','knn'])].copy()
    old_base = pd.read_csv(project_path('outputs/dual_vq/20260928_190154_268b85')/'metrics.csv')
    old_base = old_base[old_base.method.isin(['classifier','knn'])].copy()
    base_effects = paired(references,old_base,['method','seed','split'],'four features - five features',['phoneme_accuracy'])
    base_effects.to_csv(run/'historical_reference_effects.csv',index=False)
    raw = pd.read_csv(run/'raw_seed_metrics.csv')
    raw = raw[raw.split.ne('val')]
    old_raw = pd.read_csv(project_path(cfg['reference_run'])/'raw_seed_metrics.csv')
    old_raw = old_raw[(old_raw.view=='pooled')&old_raw.split.ne('val')]
    raw_effects = paired(raw,old_raw,['method','seed','split'],'four features - five features',
                         ['phoneme_accuracy','phoneme_hungarian_accuracy','phi_phi_nmi','phi_phi_ami','phi_phi_ari'])
    raw_effects.to_csv(run/'historical_raw_effects.csv',index=False)
    fits = json.loads((run/'raw_fits.json').read_text())
    diagnostics = pd.DataFrame([{**{k:r[k] for k in ('method','seed')},**f} for r in fits for f in r['fits']])
    diagnostics.to_csv(run/'raw_fit_diagnostics.csv',index=False)
    groups = ['method','k_phi','lambda_s','split']
    allmetrics = ['phoneme_accuracy','phi_phi_nmi','phi_phi_ami','phi_phi_ari','phi_phi_purity','phi_s_nmi','s_phi_nmi','s_s_nmi',
                  'phi_active_codes','s_active_codes','phi_perplexity','s_perplexity','mse']
    disc.groupby(groups)[allmetrics].agg(['mean','std']).to_csv(run/'summary.csv')
    figs = run/'figures'; figs.mkdir(exist_ok=True)
    for name,columns in [('performance',['phoneme_accuracy','phi_phi_nmi','phi_phi_ari']),
                         ('factorization',['phi_phi_nmi','phi_s_nmi','s_phi_nmi','s_s_nmi']),
                         ('usage',['phi_active_codes','s_active_codes','phi_perplexity','s_perplexity'])]:
        fig,axes = plt.subplots(2,len(columns),figsize=(5*len(columns),7),squeeze=False)
        for row,part in enumerate(('test_seen','test_unseen')):
            for col,metric in enumerate(columns):
                ax = axes[row,col]
                for method,weight,label in [('vq',0,'VQ lambda=0'),('vq',1,'VQ lambda=1'),('gmvae',-1,'GMVAE unsupervised')]:
                    g = disc[(disc.method==method)&(disc.lambda_s==weight)&(disc.split==part)].groupby('k_phi')[metric]
                    ax.errorbar([0,1],g.mean().reindex([16,64]),yerr=g.std().reindex([16,64]),marker='o',capsize=3,label=label)
                ax.set(xticks=[0,1],xticklabels=['16','64'],xlabel='States per factor',title=part+': '+metric)
                ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs/(name+'.png'),dpi=140); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,part in zip(axes,('test_seen','test_unseen')):
        for weight in (0,1):
            values = [d[(d.lambda_s==weight)&(d.split==part)].phoneme_accuracy for d in (old,now)]
            ax.errorbar([0,1],[v.mean() for v in values],yerr=[v.std() for v in values],marker='o',capsize=3,label=f'VQ64 lambda={weight}')
        ax.set(xticks=[0,1],xticklabels=['5 features','4 features'],ylabel='Frozen majority accuracy',title=part)
        ax.legend()
    fig.tight_layout(); fig.savefig(figs/'feature_removal.png',dpi=140); plt.close(fig)
    # Per-trial numerical diagnostics exist for all seeds; show first-seed examples.
    trained = pd.DataFrame(json.loads((run/'trained.json').read_text()))
    for _,r in trained[(trained.seed==42)&trained.method.isin(['vq','gmvae'])].iterrows():
        directory = run/'trials'/r.trial
        history = pd.read_csv(directory/'history.csv')
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        for c in history:
            if c!='epoch':
                ax = axes[0] if c.startswith('train') else axes[1]
                ax.plot(history.epoch,history[c],label=c)
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.legend(fontsize=6)
        fig.suptitle(r.trial); fig.tight_layout(); fig.savefig(figs/(r.trial+'_history.png'),dpi=120); plt.close(fig)
        fig,axes = plt.subplots(2,2,figsize=(10,7))
        for row,part in enumerate(('test_seen','test_unseen')):
            for col,b in enumerate(('phi','s')):
                usage = json.loads((directory/f'{part}_{b}_usage.json').read_text())['counts']
                axes[row,col].bar(np.arange(len(usage)),usage)
                axes[row,col].set(title=f'{part}: {b}',xlabel='Discrete state',ylabel='Tokens')
        fig.tight_layout(); fig.savefig(figs/(r.trial+'_occupancy.png'),dpi=120); plt.close(fig)
    parameters = trained[trained.method.ne('knn')].groupby(['method','k_phi']).parameter_count.first().reset_index()
    sections = [
        '# Four-feature VQ-VAE versus original GMVAE',
        '## Design',
        'Only F1, F2, F3 and duration enter input, reconstruction, cleaning and preprocessing. No F0 or missing-F0 indicator is loaded. '
        'The actual previous split manifest is loaded and copied byte-for-byte; speaker identities and token membership are unchanged. '
        'Training-only median imputation and standardization match the corresponding four old feature statistics. '
        'The manifest stores membership, not the old randomized array ordering: new arrays use manifest/source-row order, identically for all new conditions. '
        'Historical same-seed comparisons therefore do not imply identical initialization or minibatch sequences across feature dimensions.',
        md_table(pd.DataFrame(data['speaker_counts']).fillna(0).astype(int).reset_index(names='speaker')),
        md_table(pd.DataFrame(data['phoneme_counts']).fillna(0).astype(int).reset_index(names='vowel')),
        '## Models and training',
        'VQ: two 4→128 ReLU→128 ReLU→8 encoders; concatenated quantized vectors →128 ReLU→128 ReLU→4 decoder; '
        'linear 8→10 speaker head on z_s. K=16/16 or 64/64, lambda=0 or 1; seeds 42–46. '
        'Loss = feature-mean MSE + sum over branches of [mean squared codebook error + 0.25 mean squared commitment error] + lambda CE. '
        'Gradient-updated embeddings, standard straight-through h+(e-h).detach(), no EMA/reset. CE gradients enter encoder/head, not embeddings directly. '
        'Ten continuous-AE epochs and train-only 10-restart K-means initialize each branch; Adam .001, batch512, clip10, 80 fixed epochs, final checkpoint. '
        'Initial weights and minibatch sequences match across lambda; neural initial weights also match across K. No phoneme labels enter VQ training.',
        'GMVAE is the original TwoFactorVaDE, not the shared-w or neural-grid variant: 4→128 ReLU→128 ReLU encoder, '
        '8D diagonal-Gaussian z, symmetric nonlinear decoder; independent uniform categorical priors, with K×K diagonal Gaussian components. '
        'Component mean = global + centered phoneme effect + centered speaker effect + double-centered interaction. '
        'K=16 means 256 joint components; K=64 means 4096, unlike VQ with 2K embeddings. '
        'The original loss is reconstruction Gaussian NLL (fixed variance .1) + responsibility-weighted Gaussian KL + categorical KL + 10×mean squared centered interaction. '
        'Responsibilities are averaged over four posterior samples during training; evaluation uses 32 and marginal argmax per factor. '
        'Twenty AE epochs, sequential residual K-means initialization (10 restarts), Adam .001, batch512, clip10, up to100 epochs; '
        'validation original-objective early stopping (patience20, min_delta1e-4), best checkpoint. No speaker/phoneme supervision. '
        'All statistical settings are unchanged. Only Gaussian quadratic computations are algebraically reassociated into matrix products to avoid enormous broadcast tensors; '
        'double-precision loss/gradient equivalence tests pass. Execution uses deterministic CUDA on RTX3060 rather than the old CPU; finite-precision trajectories can differ. '
        'Reconstruction MSE uses decoder(posterior mean), as in the preceding baseline; it is not the MC training NLL.',
        'Supervised references retain the earlier 15-NN and nonlinear dual-head classifier with two128-unit ReLU hidden layers and 8D bottleneck. '
        'Classifier optimization/validation stopping follows the old defaults above, without AE pretraining. '
        'The classifier uses both true phoneme and familiar-speaker targets; both references are supervised, not equivalent clustering methods. '
        '15-NN is deterministic and fitted once, not presented as five independent seeds.',
        md_table(parameters),
        '## Compact comparison: mean ± sample SD',
        'Lambda -1 means not applicable (original unsupervised GMVAE). Accuracy is percent. '
        'Every discrete state maps to its training-majority vowel; mappings are frozen before validation/test evaluation. '
        'Unused training states fall back to the training-majority label. Purity is descriptive test contingency purity, not a test-fitted prediction mapping. '
        'All conditions are reported; no best test lambda is selected.',
        table(disc,groups,allmetrics,['phoneme_accuracy']),
        'Fraction assigned to states unused in training (these use the documented majority fallback):',
        table(disc,groups,['phi_unmapped_fraction','s_unmapped_fraction']),
        'Per-feature standardized reconstruction MSE (within this four-feature experiment only):',
        table(disc,groups,['mse_f1_hz','mse_f2_hz','mse_f3_hz','mse_duration_s']),
        'Speaker classification and training-fitted speaker-code accuracy are defined only for familiar speakers:',
        table(disc[disc.split.eq('test_seen')],['method','k_phi','lambda_s'],
              ['speaker_classifier_accuracy','speaker_code_accuracy'],['speaker_classifier_accuracy','speaker_code_accuracy']),
        table(references,['method','split'],['phoneme_accuracy','speaker_classifier_accuracy'],['phoneme_accuracy','speaker_classifier_accuracy']),
        '![Performance](figures/performance.png)',
        '![Factorization](figures/factorization.png)',
        '![Usage](figures/usage.png)',
        '## Paired effects',
        'All contrasts below subtract matched seeds. Accuracy differences are percentage points; full per-seed values are in paired_effects.csv. '
        'The unsupervised VQ versus GMVAE comparison is free of explicit speaker supervision but retains different latent geometry, capacity scaling, objectives and checkpoint rules. '
        'Supervised VQ versus GMVAE is a comparison of complete learning systems, not a pure architectural effect.',
        table(effects,['comparison','split'],METRICS,['phoneme_accuracy']),
        '## Historical five-feature comparisons',
        'These use saved historical results, not any F0-bearing input in the new experiment. No cross-dimensional raw MSE difference is reported. '
        'Previous classifier results exist for seeds42–44 only; paired classifier contrasts therefore use those three common seeds.',
        table(historical,['lambda_s','split'],METRICS[:-1],['phoneme_accuracy']),
        table(base_effects,['method','split'],['phoneme_accuracy'],['phoneme_accuracy']),
        '![Feature removal](figures/feature_removal.png)',
        '## Pooled raw clusterability control',
        'Five clusters; K-means10 restarts and full-covariance GMM5 restarts, max300 iterations, reg_covar1e-5. '
        'Native unlabeled fit objectives choose restarts; training-fitted Hungarian and many-to-one mappings are frozen. No repeated F0 association analysis.',
        f'Convergence: {int((~diagnostics.converged).sum())}/{len(diagnostics)} final raw fits flagged nonconverged; full diagnostics are saved in raw_fit_diagnostics.csv.',
        table(raw,['method','split'],['phoneme_hungarian_accuracy','phoneme_accuracy','phi_phi_nmi','phi_phi_ami','phi_phi_ari'],['phoneme_hungarian_accuracy','phoneme_accuracy']),
        'Changes versus historical pooled five-feature fits:',
        table(raw_effects,['method','split'],['phoneme_hungarian_accuracy','phoneme_accuracy','phi_phi_nmi','phi_phi_ari'],['phoneme_hungarian_accuracy','phoneme_accuracy']),
        '## Interpretation and limits',
    ]
    for k in (16,64):
        g = effects[effects.comparison.eq(f'VQ K{k}: lambda1 - lambda0')].groupby('split')[['phoneme_accuracy','phi_phi_nmi','phi_s_nmi']].mean()
        sections.append(f'At VQ K={k}, speaker supervision changes phoneme accuracy by {100*g.loc["test_seen","phoneme_accuracy"]:+.2f} pp seen / '
                        f'{100*g.loc["test_unseen","phoneme_accuracy"]:+.2f} pp unseen. Phoneme NMI changes by '
                        f'{g.loc["test_seen","phi_phi_nmi"]:+.3f} / {g.loc["test_unseen","phi_phi_nmi"]:+.3f}; '
                        f'speaker contamination NMI changes by {g.loc["test_seen","phi_s_nmi"]:+.3f} / {g.loc["test_unseen","phi_s_nmi"]:+.3f}.')
    sections += [
        'More active states, lower MSE and higher majority accuracy are not by themselves categorical discovery. '
        'NMI/AMI/ARI depend on partition granularity; examine them jointly with both cross-factor associations. '
        'GMVAE can reconstruct through continuous z without its intended categories aligning to labels, unlike VQ reconstruction through discrete vectors. '
        'The GMVAE factor names are intended roles and its residual initialization is asymmetric; no test-based axis swapping is performed. '
        'Five optimization seeds on one fixed speaker sample do not estimate speaker-population uncertainty; familiar-speaker splits are tokens, not held-out utterances. '
        'Both test sets were examined in preceding experiments: these are controlled exploratory comparisons, not a fresh confirmatory holdout.',
        '## Reproduction',
        '`python src/train_four_feature.py --config src/configs/four_feature.py`; add `--resume RUN_DIRECTORY` to reuse a matching plan. '
        'The notebook src/train_four_feature.ipynb runs the same CLI and report. All configurations, seeds, source snapshots, split manifest, preprocessing, '
        'model checkpoints, curves, predictions, frozen mappings, contingencies, occupancy and per-feature errors are retained. '
        'GMVAE device is explicit in the config. First-seed curves/occupancy plots are shown as examples; numeric histories/diagnostics exist for every run.'
    ]
    (run/'REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print('Report:',run/'REPORT.md',flush=True)
    return run/'REPORT.md'


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('run_directory')
    make_four_report(p.parse_args().run_directory)
