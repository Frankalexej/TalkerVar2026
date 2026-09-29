"""Reports and plots for raw clusterability and VQ capacity/dose responses."""
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
from sklearn.metrics import normalized_mutual_info_score
from src.config import project_path
from src.vq_report import md_table


def table(frame,groups,metrics,percent=()):
    rows = []
    for keys,g in frame.groupby(groups,sort=True):
        if not isinstance(keys,tuple):
            keys = (keys,)
        row = dict(zip(groups,keys))
        row['seeds'] = g.seed.nunique()
        for metric in metrics:
            values = g[metric].dropna() if metric in g else pd.Series(dtype=float)
            scale = 100 if metric in percent else 1
            row[metric] = '—' if len(values)==0 else f'{values.mean()*scale:.3f}' if len(values)==1 else f'{values.mean()*scale:.3f} ± {values.std()*scale:.3f}'
        rows.append(row)
    return md_table(pd.DataFrame(rows))


def make_sensitivity_report(run):
    run = Path(run)
    if not (run/'complete.json').exists():
        raise RuntimeError('Wait for all preregistered runs to complete.')
    plan = json.loads((run/'plan.json').read_text())
    cfg = plan['config']
    data = json.loads((run/'data.json').read_text())
    frame = pd.read_csv(run/'vq_metrics.csv')
    vq = frame[frame.split != 'val']
    raw = pd.read_csv(run/'raw_seed_metrics.csv')
    raw = raw[raw.split != 'val']
    individuals = pd.read_csv(run/'raw_metrics.csv')
    individuals = individuals[(individuals.view=='individual')&(individuals.split=='test_seen')]
    paired = pd.read_csv(run/'paired_dose_effects.csv')
    paired = paired[paired.split != 'val']
    capacity = pd.read_csv(run/'paired_capacity_effects.csv')
    capacity = capacity[capacity.split != 'val']
    fits = json.loads((run/'raw_fits.json').read_text())
    diagnostics = [{**{k:r[k] for k in ('method','view','seed')},**fit} for r in fits for fit in r['fits']]
    pd.DataFrame(diagnostics).to_csv(run/'raw_fit_diagnostics.csv',index=False)
    # Post-fit diagnostic only; the feature set and every fitted model stay fixed.
    f0 = pd.read_csv(project_path(cfg['data']['csv_path']),usecols=['f0_median_hz']).iloc[:,0].to_numpy()
    missing = (~np.isfinite(f0)) | (f0 <= 0)
    missing_rows = []
    for method in ('kmeans','gmm'):
        for view in ('pooled','speaker_normalized'):
            for seed in cfg['seeds']:
                for part in (('test_seen','test_unseen') if view=='pooled' else ('test_seen',)):
                    pred = pd.read_csv(run/'raw'/f'{method}_{view}_seed_{seed}'/f'{part}_predictions.csv')
                    mask = missing[pred.source_row.to_numpy()]
                    missing_rows.append({'method':method,'view':view,'seed':seed,'split':part,
                                         'f0_missing_fraction':float(mask.mean()),
                                         'cluster_missingness_nmi':float(normalized_mutual_info_score(mask,pred.local_cluster))})
    missing_frame = pd.DataFrame(missing_rows)
    missing_frame.to_csv(run/'raw_missingness_diagnostics.csv',index=False)
    figs = run/'figures'
    figs.mkdir(exist_ok=True)
    dose = list(cfg['lambdas'])
    for filename,metrics in (('dose_response',['phoneme_accuracy','mse','speaker_classifier_accuracy']),
                             ('factor_information',['phi_phi_nmi','phi_s_nmi','s_phi_nmi','s_s_nmi']),
                             ('usage',['phi_active_codes','phi_perplexity','s_active_codes','s_perplexity'])):
        fig,axes = plt.subplots(2,len(metrics),figsize=(5*len(metrics),7),squeeze=False)
        for row,part in enumerate(('test_seen','test_unseen')):
            for col,metric in enumerate(metrics):
                ax = axes[row,col]
                if metric=='speaker_classifier_accuracy' and part=='test_unseen':
                    ax.set_axis_off()
                    ax.text(.5,.5,'Speaker classification accuracy\nis undefined for unseen identities',transform=ax.transAxes,ha='center',va='center')
                    continue
                for kp,ks in cfg['codebook_pairs']:
                    g = vq[(vq.k_phi==kp)&(vq.k_s==ks)&(vq.split==part)].groupby('lambda_s')[metric].agg(['mean','std']).reindex(dose)
                    ax.errorbar(range(len(dose)),g['mean'],yerr=g['std'],marker='o',capsize=3,label=f'{kp}/{ks} codes')
                ax.set_xticks(range(len(dose)),dose)
                ax.set(title=f'{part}: {metric}',xlabel='Speaker CE weight')
                ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs/f'{filename}.png',dpi=140); plt.close(fig)
    views = ['pooled','individual_micro','speaker_normalized']
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,['phoneme_hungarian_accuracy','phi_phi_nmi','phi_phi_ari']):
        for offset,method in enumerate(('kmeans','gmm')):
            g = raw[(raw.method==method)&(raw.split=='test_seen')].groupby('view')[metric].agg(['mean','std']).reindex(views)
            ax.bar(np.arange(3)+(offset-.5)*.35,g['mean'],yerr=g['std'],width=.35,label=method)
        ax.set_xticks(range(3),['Pooled','Individual\n(weighted)','Speaker\nnormalized'])
        ax.set(title=metric); ax.legend()
    fig.tight_layout(); fig.savefig(figs/'raw_clusterability.png',dpi=150); plt.close(fig)
    fig,ax = plt.subplots(figsize=(12,4))
    for offset,method in enumerate(('kmeans','gmm')):
        g = individuals[individuals.method==method].groupby('speaker').phoneme_hungarian_accuracy.agg(['mean','std']).reindex(data['training_speakers'])
        ax.errorbar(np.arange(10)+(offset-.5)*.1,g['mean'],yerr=g['std'],fmt='o',capsize=3,label=method)
    ax.set_xticks(range(10),data['training_speakers'],rotation=45)
    ax.set(ylabel='Frozen Hungarian phoneme accuracy',title='Independent familiar-speaker models'); ax.legend()
    fig.tight_layout(); fig.savefig(figs/'individual_speakers.png',dpi=150); plt.close(fig)
    # One prespecified seed per setting; all histories/contingencies remain saved.
    seed = cfg['seeds'][0]
    for kp,ks in cfg['codebook_pairs']:
        for weight in dose:
            name = f'K{kp}_{ks}_lambda_{weight:g}_seed_{seed}'
            directory = run/'vq'/name
            history = pd.read_csv(directory/'history.csv')
            fig,axes = plt.subplots(2,3,figsize=(13,7))
            for ax,metric in zip(axes.flat,['reconstruction','vq_phi','vq_s','speaker_ce','perplexity_phi','perplexity_s']):
                for part in ('train','val'):
                    ax.plot(history.epoch,history[f'{part}_{metric}'],label=part)
                ax.set(title=metric,xlabel='Epoch'); ax.legend()
            fig.suptitle(name); fig.tight_layout(); fig.savefig(figs/f'{name}_learning.png',dpi=110); plt.close(fig)
            fig,axes = plt.subplots(2,4,figsize=(17,9))
            for row,part in enumerate(('test_seen','test_unseen')):
                for col,(b,factor) in enumerate((('phi','phi'),('phi','s'),('s','phi'),('s','s'))):
                    a = pd.read_csv(directory/f'{part}_{b}_by_{factor}.csv',index_col=0)
                    ax = axes[row,col]
                    im = ax.imshow(a,aspect='auto',cmap='Blues')
                    names = [data['phonemes'][int(c)] if factor=='phi' else data['speaker_order'][int(c)] for c in a.columns]
                    ax.set_xticks(range(len(names)),names,rotation=65)
                    ax.set(title=f'{part}: {b} × {factor}',ylabel='VQ code')
                    fig.colorbar(im,ax=ax,shrink=.65)
            fig.suptitle(name); fig.tight_layout(); fig.savefig(figs/f'{name}_contingencies.png',dpi=110); plt.close(fig)
    counts = pd.DataFrame(data['speaker_counts']).fillna(0).astype(int).reindex(data['speaker_order'])
    counts.index.name = 'speaker'
    initial = json.loads((run/'vq'/f'K64_64_lambda_0_seed_{seed}'/'done.json').read_text())
    sections = [
        '# Larger VQ codebooks, quantized speaker supervision and raw clusterability',
        '## Design and interpretation of the attached analysis',
        'The attached analysis usefully separates supervised discrimination, density clustering and discrete compression. '
        'These experiments test raw clusterability and a post-quantization speaker-supervision dose response. '
        'Low scores from two clustering algorithms do not establish a universal ceiling for unsupervised learning; '
        'high code usage and decoder dependence do not establish semantic factorization. Context pooling, probes, '
        'phoneme-supervised oracles, F0 ablations and hierarchical mixtures were not part of these two requested tasks.',
        f"Run `{run.name}`; seed set {cfg['seeds']}. Same 13 speakers, exact token split and training-only preprocessing as "
        f"`{cfg['reference_run']}` (verified against its metadata and full split manifest). "
        'Input/target is median F0, F1, F2, F3 and duration; nonpositive/nonfinite values are median-imputed using training observations, '
        'then standardized with training statistics. All five vowel categories occur in all splits. Familiar-speaker tokens are not grouped by utterance.',
        md_table(counts.reset_index()),
        '## VQ protocol',
        f"Two encoders: 5 → 128 ReLU → 128 ReLU → 8; decoder: 16 → 128 ReLU → 128 ReLU → 5. "
        f"Codebook pairs are {cfg['codebook_pairs']}. The larger model has {initial['parameter_count']:,} parameters. "
        'Speaker classifier: quantized z_s (8) → 10 familiar-speaker logits. There is no CE on h_s and no phoneme supervision.',
        '`L = mean((x-xhat)^2) + Σ_b [mean((sg(h_b)-e_b)^2) + 0.25 mean((h_b-sg(e_b))^2)] + lambda_s CE(C_s(z_s), s)`',
        f"All lambda values {dose} are reported, with {cfg['epochs']} epochs and the final checkpoint as the primary outcome. "
        'There is no reconstruction-based choice of weight or epoch. Each model uses ten continuous-autoencoder pretraining epochs, '
        'separate training-only K-means initialization (10 restarts), Adam LR 0.001, batch 512 and gradient clipping 10. '
        'Within a capacity, initial states and minibatch orders match exactly across lambda; initial neural parameters and minibatches also match across capacities. '
        'Standard straight-through `h + (e-h).detach()` sends CE gradients to the encoder and classifier; codebook embeddings receive their gradients from the VQ codebook loss, not directly from CE. No EMA or dead-code resets.',
        'For overcomplete codes, phoneme/speaker mapping assigns each code its training-majority category; multiple codes can map to one label. '
        'Codes unused during training fall back to the training-majority label, with their test fraction reported. Mappings are frozen on both tests. '
        'One-to-one Hungarian scores are additionally saved for the 5/10 control only. Many-to-one accuracy is not numerically interchangeable with the old Hungarian accuracy. '
        'All capacities use the same many-to-one metric in the comparison below. More codes make purity and majority mapping more flexible; NMI/ARI/AMI also depend on partition granularity and may penalize splitting one vowel across several pure codes. '
        'Chance-adjusted MI (AMI) is reported alongside NMI/ARI, but none makes different codebook sizes perfectly equivalent.',
        '## VQ results: all settings, mean ± sample SD',
        'Accuracy columns are percentages. The lambda-zero head is untrained and not evaluated; speaker classifier/code accuracy is undefined for unseen identities.',
        table(vq,['k_phi','k_s','lambda_s','split'],['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy','mse'],
              ['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy']),
        '![Dose response](figures/dose_response.png)',
        table(vq,['k_phi','k_s','lambda_s','split'],['phi_phi_nmi','phi_phi_ari','phi_phi_ami','phi_s_nmi','s_phi_nmi','s_s_nmi']),
        '![Factor information](figures/factor_information.png)',
        table(vq,['k_phi','k_s','lambda_s','split'],['phi_active_codes','phi_perplexity','s_active_codes','s_perplexity','phi_unmapped_fraction','s_unmapped_fraction']),
        '![Usage](figures/usage.png)',
        '## Paired effects',
        'Within-capacity differences are lambda minus zero, paired by seed; accuracy differences are percentage points. '
        'All per-seed effects are in `paired_dose_effects.csv`; capacity effects (64/64 minus 5/10 at the same lambda/seed) are in `paired_capacity_effects.csv`.',
        table(paired,['k_phi','k_s','lambda_s','split'],['phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi','s_s_nmi','mse'],['phoneme_accuracy']),
        'Paired capacity effects:',
        table(capacity,['lambda_s','split'],['phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi','mse'],['phoneme_accuracy']),
        '## Raw feature clusterability',
        f"K-means: K=5, {cfg['kmeans_n_init']} restarts. GMM: five full-covariance components, {cfg['gmm_n_init']} restarts, "
        f"maximum {cfg['gmm_max_iter']} iterations, covariance regularizer {cfg['gmm_reg_covar']}. "
        'Restarts are chosen by the native unlabeled objective, not by phoneme accuracy. All fits use training observations only. '
        'Train-fitted Hungarian mappings and majority mappings are both frozen before evaluation.',
        'Pooled models use all ten familiar speakers. Individual models fit each familiar speaker separately; their cluster indices have independent semantics. '
        'Individual-micro metrics are token-weighted averages of per-speaker scores; macro metrics weight speakers equally. '
        'NMI/ARI are averaged within speaker and never computed by pooling arbitrary local cluster IDs. '
        'The individual condition also has more total model capacity (ten separate five-cluster models) than the pooled five-cluster model; its improvement cannot be attributed to speaker information alone. '
        'Speaker normalization subtracts each speaker’s training mean and divides by its training SD on the five already imputed/global-standardized features. '
        'This uses known speaker membership as a diagnostic. Its statistics are frozen for validation and familiar-speaker test. '
        'Only pooled models are evaluated on unfamiliar speakers; estimating normalization or fitting separate models for those speakers would require a different adaptation protocol.',
        table(raw,['method','view','split'],['phoneme_hungarian_accuracy','phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_phi_ami'],
              ['phoneme_hungarian_accuracy','phoneme_accuracy']),
        '![Raw clusterability](figures/raw_clusterability.png)',
        table(individuals,['method','speaker'],['phoneme_hungarian_accuracy','phi_phi_nmi','phi_phi_ari'],['phoneme_hungarian_accuracy']),
        '![Individual speakers](figures/individual_speakers.png)',
        f"Convergence: {sum(not r['converged'] for r in diagnostics)}/{len(diagnostics)} final fits flagged nonconverged. "
        'Per-fit iterations, objective values and captured convergence warnings are recorded in `raw_fit_diagnostics.csv`.',
        'Additional post-fit diagnostic: association of existing cluster assignments with the missing/invalid-F0 indicator. '
        'No features or fits were changed for this calculation.',
        table(missing_frame,['method','view','split'],['f0_missing_fraction','cluster_missingness_nmi']),
        '## Interpretation and limitations',
    ]
    for method in ('kmeans','gmm'):
        g = raw[(raw.method==method)&(raw.split=='test_seen')].groupby('view').phoneme_hungarian_accuracy.mean()
        sections.append(f"{method}: familiar-speaker Hungarian accuracy is {100*g['pooled']:.2f}% pooled, "
                        f"{100*g['individual_micro']:.2f}% with individual-speaker models, and {100*g['speaker_normalized']:.2f}% "
                        'after speaker normalization. These contrasts test location/scale interference and model fit; they do not prove an intrinsic clusterability limit.')
    for method in ('kmeans','gmm'):
        values = missing_frame[(missing_frame.method==method)&(missing_frame.split=='test_seen')].groupby('view').cluster_missingness_nmi.mean()
        sections.append(f"{method} cluster/F0-missingness NMI is {values['pooled']:.3f} pooled and {values['speaker_normalized']:.3f} after speaker normalization. "
                        'This is an association, not an F0-ablation result. In particular, speaker normalization moves the single global imputation value '
                        'to speaker-specific standardized values, so normalization can alter the imputation artifact as well as speaker location/scale variation.')
    for kp,ks in cfg['codebook_pairs']:
        for part in ('test_seen','test_unseen'):
            g = vq[(vq.k_phi==kp)&(vq.k_s==ks)&(vq.split==part)]
            values = g.groupby('lambda_s')[['phoneme_accuracy','phi_phi_nmi','phi_s_nmi','s_s_nmi','mse']].mean()
            sections.append(f"For {kp}/{ks} codes on {part.replace('test_','')} speakers, mean phoneme accuracy across lambda {dose} is "
                            + ', '.join(f'{100*values.loc[w,"phoneme_accuracy"]:.2f}%' for w in dose)
                            + '; speaker-code NMI is '+', '.join(f'{values.loc[w,"s_s_nmi"]:.3f}' for w in dose)+'. '
                            'The complete dose response, rather than its best test point, is the inferential result.')
    sections.extend([
        'A larger codebook relaxes acoustic compression and also makes training-label majority mapping more flexible. '
        'An accuracy gain alone therefore does not establish improved factor separation. Check phoneme NMI/ARI/AMI, '
        'speaker contamination in phoneme codes, and phoneme leakage into speaker codes together. '
        'The small-code control shares the new CE placement and fixed final-epoch protocol; comparisons with the older h_s-supervised run are historical, '
        'because that run selected checkpoints by validation reconstruction.',
        'Median-imputed F0 creates a point mass (32.6% of training tokens have missing/invalid F0). '
        'GMM density components can model that feature of the data rather than vowel identities; this experiment retains the original cleaning convention. '
        'Per-speaker normalization removes only location/scale differences and uses speaker information explicitly. '
        'One fixed speaker selection and five optimization seeds do not estimate variation across independent speaker samples. '
        'No representation probe, F0-removal control or speaker-context intervention was run here.',
        '## Reproduction',
        '`python src/train_sensitivity.py --config src/configs/vq_sensitivity.py`; resume with `--resume RUN_DIRECTORY`. '
        'The executed notebook is `src/train_vq_sensitivity.ipynb`. All source snapshots, config/seeds, preprocessing, models, mappings, '
        'histories, predictions, per-feature MSE, ablations and metrics are saved in this run. '
        'Regenerate this report with `python src/sensitivity_report.py RUN_DIRECTORY`. '
        'First-seed learning curves and four-way contingency plots are in `figures/`; every seed’s numeric diagnostics are retained in its trial directory.'
    ])
    (run/'REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print('Report:',run/'REPORT.md',flush=True)
    return run/'REPORT.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory')
    make_sensitivity_report(parser.parse_args().run_directory)
