"""Create reproducible diagnostic figures and a compact technical report."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def md_table(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def summarize(frame, metrics, percent=()):
    rows = []
    for (method, split), group in frame.groupby(['method','split'], sort=False):
        row = {'Method':method, 'Test':split.replace('test_',''), 'n':len(group)}
        for metric in metrics:
            a = group[metric].dropna() if metric in group else pd.Series(dtype=float)
            scale = 100 if metric in percent else 1
            row[metric] = ('—' if len(a) == 0 else f'{a.mean()*scale:.3f}' if len(a) == 1 else
                           f'{a.mean()*scale:.3f} ± {a.std(ddof=1)*scale:.3f}')
        rows.append(row)
    return md_table(pd.DataFrame(rows))


def trial_figures(directory, metadata):
    history = pd.read_csv(directory/'history.csv')
    plots = directory/'figures'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,3,figsize=(14,7))
    for ax, metric in zip(axes.flat, ('reconstruction','vq_phi','vq_s','speaker_ce','perplexity_phi','perplexity_s')):
        for part in ('train','val'):
            ax.plot(history.epoch, history[f'{part}_{metric}'], label=part)
        ax.set(title=metric, xlabel='Epoch')
        ax.legend()
    fig.suptitle(directory.name+' — training curves')
    fig.tight_layout(); fig.savefig(plots/'learning.png',dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for row, part in enumerate(('test_seen','test_unseen')):
        for col, branch in enumerate(('phi','s')):
            usage = json.loads((directory/f'{part}_{branch}_usage.json').read_text())
            axes[row,col].bar(range(len(usage['counts'])), usage['counts'])
            axes[row,col].set(title=f'{part}: K_{branch}', xlabel='Code',ylabel='Tokens')
    fig.tight_layout(); fig.savefig(plots/'occupancy.png',dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(18,8))
    for row, part in enumerate(('test_seen','test_unseen')):
        for col, (branch,factor) in enumerate((('phi','phi'),('phi','s'),('s','phi'),('s','s'))):
            counts = pd.read_csv(directory/f'{part}_{branch}_by_{factor}.csv',index_col=0)
            ax = axes[row,col]
            image = ax.imshow(counts,aspect='auto',cmap='Blues')
            ax.set_xticks(range(len(counts.columns)),counts.columns,rotation=70)
            ax.set_yticks(range(len(counts)))
            ax.set(title=f'{part}: K_{branch} × {factor}',ylabel='Code')
            fig.colorbar(image,ax=ax,shrink=.7)
    fig.tight_layout(); fig.savefig(plots/'contingencies.png',dpi=130); plt.close(fig)
    train = np.load(directory/'train_representations.npz')
    pcas = {b:PCA(n_components=2).fit(train['h_'+b]) for b in ('phi','s')}
    fig, axes = plt.subplots(2,4,figsize=(18,8))
    for row, part in enumerate(('test_seen','test_unseen')):
        a = np.load(directory/f'{part}_representations.npz')
        ix = np.random.default_rng(2026).choice(len(a['true_phoneme']), min(3500,len(a['true_phoneme'])),replace=False)
        for col, (branch,factor) in enumerate((('phi','phoneme'),('phi','speaker'),('s','phoneme'),('s','speaker'))):
            z = pcas[branch].transform(a['h_'+branch][ix])
            labels = a['true_'+factor][ix]
            ax = axes[row,col]
            for label in np.unique(labels):
                mask = labels == label
                name = metadata['phonemes'][label] if factor == 'phoneme' and label>=0 else (
                    'unknown' if factor == 'phoneme' else metadata['speaker_order'][label])
                ax.scatter(z[mask,0],z[mask,1],s=4,alpha=.45,label=name)
            ax.set(title=f'{part}: h_{branch} by {factor}',xlabel='PC1',ylabel='PC2')
            ax.legend(fontsize=6,markerscale=2,ncol=2)
    fig.tight_layout(); fig.savefig(plots/'pca.png',dpi=130); plt.close(fig)


def make_report(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir/'complete.json').exists():
        raise RuntimeError('Experiment must finish before reporting test results.')
    data = json.loads((run_dir/'data.json').read_text())
    plan = json.loads((run_dir/'plan.json').read_text())
    cfg = plan['config']
    choice = json.loads((run_dir/'selection.json').read_text())
    results = pd.read_csv(run_dir/'metrics.csv')
    paired = pd.read_csv(run_dir/'paired_differences.csv')
    vq = results[results.method.isin(['A','B'])]
    figures = run_dir/'figures'
    figures.mkdir(exist_ok=True)
    for directory in sorted((run_dir/'final').iterdir()):
        trial_figures(directory,data)
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax, metric in zip(axes,('phoneme_accuracy','phi_phi_nmi','phi_s_nmi')):
        for i, part in enumerate(('test_seen','test_unseen')):
            g = paired[paired.split == part]
            ax.scatter(np.full(len(g),i),g[metric],s=45)
            ax.plot([i-.15,i+.15],[g[metric].mean()]*2,color='black')
        ax.axhline(0,color='gray',linestyle='--')
        ax.set_xticks([0,1],['Seen','Unseen']); ax.set(title=metric,ylabel='Paired A − B')
    fig.tight_layout(); fig.savefig(figures/'paired_effects.png',dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, part in zip(axes,('test_seen','test_unseen')):
        for i, method in enumerate(('A','B')):
            g = vq[(vq.method == method)&(vq.split == part)]
            columns = ['mse_'+f for f in cfg['data']['features']]
            ax.bar(np.arange(5)+(i-.5)*.35,g[columns].mean(),width=.35,yerr=g[columns].std(),label=method)
        ax.set_xticks(range(5),['F0','F1','F2','F3','duration'])
        ax.set(title=part,ylabel='Standardized feature MSE'); ax.legend()
    fig.tight_layout(); fig.savefig(figures/'feature_reconstruction.png',dpi=150); plt.close(fig)
    speakers = pd.DataFrame(data['speaker_counts']).fillna(0).astype(int).reindex(data['speaker_order'])
    speakers.index.name = 'Speaker'
    phonemes = pd.DataFrame(data['phoneme_counts']).fillna(0).astype(int)
    phonemes.index.name = 'Vowel'
    distribution = phonemes.div(phonemes.sum(0),axis=1)*100
    train0 = json.loads((run_dir/'final'/f'A_seed_{cfg["seeds"][0]}'/'validation.json').read_text())
    sections = [
        '# Dual-codebook VQ-VAE experiment',
        '## Design and data',
        f"Run `{run_dir.name}`. Data seed {cfg['data']['seed']}; CSV SHA256 `{data['csv_sha256']}`. "
        'The original ten speakers were retained; three additional speakers were sampled without replacement from the remainder using seed 2026. '
        'Each familiar speaker was randomly split independently into 80/10/10 (rounding accounts for integer counts). '
        'This differs from the old jointly stratified token split, so every comparison below was rerun on the new split. '
        'Tokens are the split unit; utterances were not grouped. The unseen-speaker test contains all selected tokens.',
        'Exact split totals: ' + ', '.join(f'{p}={n:,}' for p,n in data['split_counts'].items()) + '.',
        md_table(speakers.reset_index()),
        'Phoneme counts (labels used only for reporting/evaluation in VQ experiments):',
        md_table(phonemes.reset_index()),
        'Phoneme percentages:', md_table(distribution.round(2).reset_index()),
        f"Categories absent from training: {data['absent_training_phonemes'] or 'none'}. "
        'Nonfinite/nonpositive features become missing; training medians impute them and training means/SDs standardize all four splits. '
        'No log transform. Missing F0 tokens are retained; reconstruction measures the imputed target as in the previous experiments. '
        'The training and unseen F0 missing/invalid counts are '
        f"{data['missing_or_invalid']['train']['f0_median_hz']} and {data['missing_or_invalid']['test_unseen']['f0_median_hz']}, respectively.",
        '## Architecture and objective',
        f"Separate encoders: 5 → hidden layers {cfg['hidden_dims']} (ReLU) → {cfg['embedding_dim']}. "
        f"Codebooks: K_phi={len(data['phonemes'])}, K_s=10, each with {cfg['embedding_dim']}-dimensional embeddings. "
        f"Decoder: concatenate {2*cfg['embedding_dim']} → hidden layers {list(reversed(cfg['hidden_dims']))} (ReLU) → 5 linear outputs. "
        f"Auxiliary head: h_s ({cfg['embedding_dim']}) → 10 logits. "
        f"Total parameters: {train0['parameter_count']:,}; the {(cfg['embedding_dim']+1)*10}-parameter classifier is present but unoptimized and unevaluated in B. "
        'Speaker identity is a CE target only. No phoneme loss or adversarial loss is used. '
        'B’s plotted CE is a diagnostic of its inactive head and does not contribute to its loss or checkpoint selection.',
        f"`L = mean((x − x_hat)^2) + Σ_b [mean((sg(h_b) − e_b)^2) + {cfg['commitment']} mean((h_b − sg(e_b))^2)] + lambda_s CE(C_s(h_s), s)`",
        'Means average over batch and coordinates (reconstruction averages over all five features). Quantization picks the nearest embedding in squared Euclidean distance. '
        'Straight-through output is `h + (e − h).detach()`: reconstruction gradients reach the encoder; detached encoder targets update codebooks through their own loss. '
        f"Codebooks use gradient descent, not EMA. Both conditions receive {cfg['pretrain_epochs']} epochs of continuous autoencoder initialization and separate training-only K-means++ "
        '(10 restarts) on each encoder output. No dead-code resets or extra regularizers. Adam learning rate 0.001; batch 512; gradient clipping 10; no weight decay; CPU, one thread per worker.',
        '## Hyperparameter selection and matched ablation',
        f"Speaker weights {cfg['lambdas']} were compared for {cfg['selection_epochs']} epochs at seeds {cfg['selection_seeds']}. "
        'Each checkpoint minimizes validation reconstruction MSE. Configuration score is validation MSE + 0.1(1 − NMI(K_s;S)) + '
        '0.05[(1 − active_phi/5) + (1 − active_s/10)], averaged over selection seeds. These weights were prespecified. '
        f"Selected lambda_s={choice['lambda_s']:g}. No phoneme identities or test metrics were used for selection. "
        f"A uses the selected weight and B uses zero. Both train for {cfg['epochs']} epochs at seeds {cfg['seeds']}, "
        'selecting minimum validation reconstruction checkpoints. Both see identical initialization and minibatch sequences, verified by hashes; selected epochs can differ. '
        'The final seeds overlap selection seeds, so the comparison is exploratory rather than an independent replication.',
        md_table(pd.read_csv(run_dir/'selection_trials.csv')[['lambda_s','seed','validation_mse','validation_speaker_nmi','selection_score']].round(5)),
        '## Main results',
        'Mean ± sample SD across five seeds for the primary A/B comparison. Accuracy columns are percentages. '
        'Phoneme and optional familiar-speaker code mappings are fitted using training labels after training and then frozen for both tests. '
        'No classifier accuracy or Hungarian speaker accuracy is defined for unseen speakers.',
        summarize(vq,['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy','mse'],
                  ['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy']),
        'Matched baseline comparison below restricts all neural models, including A/B, to the same three seeds (42, 43, 44). '
        '15-NN is deterministic and run once. Do not confuse these three-seed means with the primary five-seed A/B means above.',
        summarize(results[results.seed.isin(cfg['baseline_seeds'])],
                  ['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy','mse'],
                  ['phoneme_accuracy','speaker_classifier_accuracy','speaker_code_accuracy']),
        '15-NN and the classifier use phoneme labels in training and are supervised references. Original GMVAE and shared-w GMVAE retain their previous hyperparameters '
        '(20 pretraining epochs, up to 100 training epochs, validation-loss early stopping). Their reconstruction uses the decoder at posterior mean z; '
        'it is not a discrete-only reconstruction, unlike the VQ models. Non-generative references have no reconstruction metric.',
        'With only two hard codes and no continuous bypass, the VQ decoder can produce at most 5×10=50 distinct reconstruction vectors. '
        'The GMVAEs retain a continuous latent z, so lower reconstruction error from a GMVAE would not by itself demonstrate better categorical factorization.',
        '## Paired speaker-supervision effects (A − B)',
        'Per-seed accuracy differences are percentage points; NMI/ARI differences are on their native scale. Positive phi-phoneme metrics and negative phi-speaker NMI are the intended pattern.',
    ]
    paired_display = paired[['seed','split','phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi']].copy()
    paired_display['phoneme_accuracy'] *= 100
    sections.append(md_table(paired_display.round(4)))
    delta_rows = []
    for part, g in paired.groupby('split'):
        row = {'Test':part}
        for metric in ('phoneme_accuracy','phi_phi_nmi','phi_phi_ari','phi_s_nmi'):
            scale = 100 if metric == 'phoneme_accuracy' else 1
            row[metric] = f'{g[metric].mean()*scale:.4f} ± {g[metric].std()*scale:.4f}'
        delta_rows.append(row)
    sections.extend([md_table(pd.DataFrame(delta_rows)), '![Paired effects](figures/paired_effects.png)',
                     '## Reconstruction by feature',
                     summarize(vq,['mse_'+f for f in cfg['data']['features']]),
                     '![Feature reconstruction](figures/feature_reconstruction.png)',
                     '## Discrete organization and leakage',
                     'NMI uses arithmetic normalization; ARI is chance-adjusted. Purity is the sum of each code’s majority-label count divided by tokens. '
                     'Purity can be high with many codes, especially 10 codes for only 3 unseen identities, so read it alongside NMI/ARI. '
                     'Seen and unseen speaker metrics involve 10 versus 3 true identities; A/B comparisons within each test set are more directly interpretable than raw cross-set speaker-metric changes. '
                     'Perplexity is exp(entropy) of hard code occupancy. Usage and rare/unused code IDs are saved per split and seed.',
                     summarize(results[results.method.isin(['A','B','gmvae','gmvae_shared_w'])],
                               ['phi_phi_nmi','phi_s_nmi','s_phi_nmi','s_s_nmi']),
                     summarize(vq,['phi_phi_ari','phi_phi_purity','s_s_ari','s_s_purity']),
                     summarize(vq,['phi_active_codes','phi_perplexity','s_active_codes','s_perplexity']),
                     '## Decoder dependence and diagnostics',
                     'Branch ablations replace all quantized vectors in that branch with its training-population mean quantized vector. '
                     'Positive MSE changes indicate decoder dependence; this is a diagnostic intervention and can produce off-codebook inputs.',
                     summarize(vq,['mse','ablate_phi_mse','ablate_phi_delta','ablate_s_mse','ablate_s_delta'])])
    interpretations = []
    for part in ('test_seen','test_unseen'):
        g = paired[paired.split == part]
        interpretations.append(f"For {part.replace('test_','')} speakers, speaker supervision changed phoneme accuracy by "
                              f"{100*g.phoneme_accuracy.mean():+.2f} points (positive in {(g.phoneme_accuracy>0).sum()}/{len(g)} seeds), "
                              f"phoneme NMI by {g.phi_phi_nmi.mean():+.3f}, ARI by {g.phi_phi_ari.mean():+.3f}, and speaker contamination NMI by {g.phi_s_nmi.mean():+.3f}.")
    for condition in ('A','B'):
        seen = vq[(vq.method == condition)&(vq.split == 'test_seen')]
        unseen = vq[(vq.method == condition)&(vq.split == 'test_unseen')]
        interpretations.append(f"Condition {condition}: mean phoneme accuracy changes from {100*seen.phoneme_accuracy.mean():.2f}% familiar to "
                              f"{100*unseen.phoneme_accuracy.mean():.2f}% unfamiliar speakers; MSE changes from {seen.mse.mean():.3f} to {unseen.mse.mean():.3f}. "
                              f"Familiar phoneme-code NMI for phoneme/speaker is {seen.phi_phi_nmi.mean():.3f}/{seen.phi_s_nmi.mean():.3f}; "
                              f"speaker-code NMI for phoneme/speaker is {seen.s_phi_nmi.mean():.3f}/{seen.s_s_nmi.mean():.3f}. "
                              f"Replacing phi/s with their means raises MSE by {seen.ablate_phi_delta.mean():.3f}/{seen.ablate_s_delta.mean():.3f}.")
    a = vq[(vq.method=='A')&(vq.split=='test_seen')]
    interpretations.append(f"The supervised speaker head achieves {100*a.speaker_classifier_accuracy.mean():.2f}% familiar-speaker accuracy, "
                          f"while the speaker VQ code achieves {100*a.speaker_code_accuracy.mean():.2f}% train-mapped accuracy and NMI {a.s_s_nmi.mean():.3f}. "
                          'The head acts before quantization: good continuous classification does not guarantee a speaker-organized discrete codebook.')
    collapse = vq[(vq.phi_active_codes < 5)|(vq.s_active_codes < 10)]
    interpretations.append(f"{len(collapse)}/{len(vq)} model/test evaluations have at least one unused code. "
                          'See occupancy histograms and usage JSON for dominant/rare codes; using every code alone does not prove factorization.')
    sections.extend(['## Interpretation', *interpretations,
                     'The A/B comparison isolates the added speaker loss under the stated selection procedure. '
                     'Its conclusion applies to the selected speaker weight and this reconstruction-oriented selection criterion; '
                     'it does not rule out different effects at other weights. '
                     'Neither branch is explicitly constrained to exclude the other factor, and the decoder may use either codebook for any acoustic variation. '
                     'The two unsupervised branches in B have no guaranteed semantic identities; their names indicate intended roles. '
                     'Five optimization seeds describe initialization variability, not uncertainty across independent speaker samples. '
                     'There is one held-out speaker trio; token-level splitting among familiar speakers and median-imputed F0 also limit generalization claims.',
                     '## Saved diagnostics and reproducibility',
                     'Every final A/B seed has learning curves, occupancy histograms, all four contingency matrices, per-speaker code distributions, '
                     'and PCA plots of both pre-quantized representations colored by both factors. PCA is fitted on training representations; '
                     'plots subsample at most 3,500 test tokens with a fixed visualization seed. First-seed figures below are prespecified examples, not the best run.',
                     f"![A learning](final/A_seed_{cfg['seeds'][0]}/figures/learning.png)",
                     f"![A occupancy](final/A_seed_{cfg['seeds'][0]}/figures/occupancy.png)",
                     f"![A contingencies](final/A_seed_{cfg['seeds'][0]}/figures/contingencies.png)",
                     f"![A PCA](final/A_seed_{cfg['seeds'][0]}/figures/pca.png)",
                     f"![B contingencies](final/B_seed_{cfg['seeds'][0]}/figures/contingencies.png)",
                     'Configuration: `src/configs/dual_vq.py`; CLI: `python src/train_vq.py --config src/configs/dual_vq.py`; '
                     'resume with `--resume RUN_DIRECTORY`. Results: `metrics.csv`, `summary.csv`, `paired_differences.csv`; '
                     'exact split: `split_manifest.csv`; preprocessing: `preprocessing.joblib`; plan/source hashes: `plan.json`; '
                     'environment: `environment.json`; training code snapshot: `source_snapshot/`. '
                     'Each trial retains initialization, best checkpoint, validation record, histories and logs. Test representations and mappings are saved. '
                     'Regenerate this report/figures with `python src/vq_report.py RUN_DIRECTORY`.' ])
    (run_dir/'REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print(f'Report: {run_dir / "REPORT.md"}',flush=True)
    return run_dir/'REPORT.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    make_report(parser.parse_args().run_directory)
