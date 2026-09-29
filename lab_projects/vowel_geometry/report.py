"""Readable Markdown and self-contained HTML reports for the lab baseline."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.vq_report import md_table

VIEWS = ['phoneme_pooled','phoneme_given_speaker','speaker_pooled','speaker_given_phoneme']
LABELS = {'phoneme_pooled':'Phoneme: pooled','phoneme_given_speaker':'Phoneme within speaker',
          'speaker_pooled':'Speaker: pooled','speaker_given_phoneme':'Speaker within phoneme'}
MEASURES = ['truth_silhouette','truth_negative_fraction','shuffled_silhouette','silhouette_above_shuffle',
            'truth_davies_bouldin','truth_calinski_harabasz','truth_between_fraction',
            'kmeans_silhouette','kmeans_davies_bouldin','kmeans_calinski_harabasz','ari','nmi','ami',
            'homogeneity','completeness','purity']
DISPLAY = {'truth_silhouette':'Silhouette','shuffled_silhouette':'Shuffled silhouette','silhouette_above_shuffle':'Difference vs shuffled',
           'truth_negative_fraction':'Negative share','truth_davies_bouldin':'DB (lower better)',
           'truth_calinski_harabasz':'CH (higher better)','truth_between_fraction':'Between / total variance',
           'kmeans_silhouette':'Cluster silhouette','kmeans_davies_bouldin':'Cluster DB','kmeans_calinski_harabasz':'Cluster CH',
           'ari':'ARI','nmi':'NMI','ami':'AMI','homogeneity':'Homogeneity','completeness':'Completeness','purity':'Purity',
           'n':'Tokens','group':'Group','view':'Analysis','mean':'Mean','repeat_sd':'Repeat SD',
           'between_group_sd':'Group SD','minimum_group_mean':'Group minimum','maximum_group_mean':'Group maximum','metric':'Metric'}


def aggregate(frame):
    # First summarize seeds within each subgroup. Group variation must not be
    # mislabeled as seed uncertainty or inflated into independent replicates.
    groups = frame.groupby(['view','group'],sort=False)[MEASURES+['n','k','n_fit','n_query']].mean().reset_index()
    rows = []
    for view in VIEWS:
        g = groups[groups.view.eq(view)]
        d = frame[frame.view.eq(view)]
        for weighting in ('macro','token_weighted'):
            weights = np.ones(len(g)) if weighting=='macro' else g.n.to_numpy()
            weights = weights/weights.sum()
            for metric in MEASURES:
                seed_means = []
                for _,part in d.groupby('seed'):
                    seed_weights = np.ones(len(part)) if weighting=='macro' else part.n.to_numpy()
                    seed_means.append(np.average(part[metric],weights=seed_weights))
                rows.append({'view':view,'weighting':weighting,'metric':metric,'mean':float(np.dot(weights,g[metric])),
                             'repeat_sd':float(np.std(seed_means,ddof=1)) if len(seed_means)>1 else None,
                             'between_group_sd':float(g[metric].std()) if len(g)>1 else None,
                             'minimum_group_mean':float(g[metric].min()),'maximum_group_mean':float(g[metric].max()),
                             'n_groups':len(g),'n_tokens':int(g.n.sum())})
    return groups,pd.DataFrame(rows)


def make_report(run):
    run = Path(run)
    assert (run/'complete.json').exists()
    cfg = json.loads((run/'plan.json').read_text())['config']
    quality = json.loads((run/'data_quality.json').read_text())
    frame = pd.read_csv(run/'metrics.csv',dtype={'group':str})
    groups,summary = aggregate(frame)
    groups.to_csv(run/'group_summary.csv',index=False)
    summary.to_csv(run/'summary.csv',index=False)
    pivot = summary[summary.weighting.eq('macro')].pivot(index='view',columns='metric',values='mean').reindex(VIEWS)
    weighted = summary[summary.weighting.eq('token_weighted')].pivot(index='view',columns='metric',values='mean').reindex(VIEWS)
    figures = run/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    short = ['Phoneme\npooled','Phoneme\nwithin speaker','Speaker\npooled','Speaker\nwithin phoneme']
    fig,axes = plt.subplots(1,3,figsize=(15,4.4))
    for ax,metric,title in zip(axes,['truth_silhouette','silhouette_above_shuffle','truth_between_fraction'],
                              ['Ground-truth silhouette','Silhouette above shuffled labels','Between-category variance fraction']):
        vals = pivot[metric].to_numpy()
        ax.bar(np.arange(4),vals,color=['#26547c','#4d9bca','#b46a20','#dea65b'])
        ax.axhline(0,color='#555555',linewidth=.8)
        ax.set(xticks=np.arange(4),xticklabels=short,title=title)
        for i,v in enumerate(vals):
            ax.annotate(f'{v:.3f}',(i,v),xytext=(0,5 if v>=0 else -14),textcoords='offset points',ha='center',fontsize=10)
        ax.margins(y=.2)
    fig.tight_layout(); fig.savefig(figures/'ground_truth.png',dpi=160); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4.4))
    for ax,metric in zip(axes,['ari','nmi','ami']):
        ax.bar(np.arange(4),pivot[metric],color=['#26547c','#4d9bca','#b46a20','#dea65b'])
        ax.set(xticks=np.arange(4),xticklabels=short,title=metric.upper()+': k-means versus category labels')
        for i,v in enumerate(pivot[metric]):
            ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
        ax.margins(y=.2)
    fig.tight_layout(); fig.savefig(figures/'cluster_agreement.png',dpi=160); plt.close(fig)
    local = groups[groups.view.eq('phoneme_given_speaker')].sort_values('truth_silhouette')
    fig,axes = plt.subplots(1,2,figsize=(13,4.4))
    axes[0].hist(local.truth_silhouette,bins=15,color='#4d9bca',edgecolor='white')
    axes[0].axvline(pivot.loc['phoneme_pooled','truth_silhouette'],color='#b46a20',label='Pooled phoneme')
    axes[0].set(xlabel='Ground-truth phoneme silhouette within each speaker',ylabel='Number of speakers'); axes[0].legend()
    axes[1].scatter(local.truth_silhouette,local.ari,s=25,color='#26547c')
    axes[1].set(xlabel='Ground-truth silhouette',ylabel='K-means ARI',title='Each point is one speaker')
    fig.tight_layout(); fig.savefig(figures/'speaker_variation.png',dpi=160); plt.close(fig)
    local_p = groups[groups.view.eq('speaker_given_phoneme')].sort_values('group')
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(local_p.group,local_p.truth_silhouette,color='#dea65b'); axes[0].set(title='Speaker silhouette within each vowel',xlabel='Vowel')
    axes[1].bar(local_p.group,local_p.ari,color='#b46a20'); axes[1].set(title='Speaker k-means ARI within each vowel',xlabel='Vowel')
    fig.tight_layout(); fig.savefig(figures/'phoneme_variation.png',dpi=160); plt.close(fig)
    # A common diagnostic projection, not a substitute for full-dimensional metrics.
    x = np.load(run/'standardized.npy',mmap_mode='r')
    pca = PCA(n_components=2).fit(x)
    sample = np.sort(np.random.default_rng(2026).choice(len(x),15000,replace=False))
    z = pca.transform(x[sample]); yp = np.load(run/'phoneme.npy')[sample]
    fig,axes = plt.subplots(1,2,figsize=(12,4.6))
    for i,label in enumerate(quality['phonemes']):
        ix = yp==i
        axes[0].scatter(z[ix,0],z[ix,1],s=4,alpha=.35,label=label,rasterized=True)
    axes[0].legend(markerscale=3); axes[0].set(title='Pooled vowels: common PCA projection')
    ys = np.load(run/'speaker.npy')
    centers = np.array([x[ys==i].mean(0) for i in range(quality['n_speakers'])])
    cz = pca.transform(centers)
    axes[1].scatter(cz[:,0],cz[:,1],s=18,color='#b46a20'); axes[1].set(title='98 speaker centroids (not token clusters)')
    for ax in axes:
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)')
    fig.tight_layout(); fig.savefig(figures/'pca.png',dpi=160); plt.close(fig)
    pd.DataFrame(pca.components_,columns=cfg['features'],index=['PC1','PC2']).to_csv(run/'pca_loadings.csv')
    chunks=[]
    def prose(title,text): chunks.append(('text',title,text))
    def tab(title,data): chunks.append(('table',title,data))
    def plot(name): chunks.append(('image',name,None))
    def formatted(p,cols):
        d = p[cols].copy().round(4); d.index=[LABELS[i] for i in d.index]
        return d.reset_index(names='Analysis').rename(columns=DISPLAY)
    prose('Dataset separation baseline',
          f"All {quality['n_tokens']:,} tokens, {quality['n_speakers']} speakers and {quality['n_phonemes']} vowel categories. "
          f"Features: {', '.join(cfg['features'])}. Formants/F0 are in Hz; duration is in seconds. This is the raw-feature baseline for a separate lab project, "
          'not a trained-model comparison or a temporal acoustic-trajectory analysis. No previous model/split files are changed.')
    prose('How to read the four views',
          '(1) Phoneme pooled: vowel categories across all speakers. '
          '(2) Phoneme within speaker: evaluate vowel separation separately for each of 98 speakers, then average. '
          '(3) Speaker pooled: 98 speaker categories across all vowels. '
          '(4) Speaker within phoneme: evaluate speaker separation separately in each of five vowels, then average. '
          'Primary conditional results are equal-weight subgroup means. Token-weighted results are supplied separately. '
          'Conditional ARI/NMI are averages of separate local comparisons, never scores on concatenated arbitrary cluster IDs.')
    prose('Main findings',
          f"Phoneme silhouette changes from {pivot.loc['phoneme_pooled','truth_silhouette']:.4f} pooled to "
          f"{pivot.loc['phoneme_given_speaker','truth_silhouette']:.4f} within speaker; k-means ARI changes from "
          f"{pivot.loc['phoneme_pooled','ari']:.4f} to {pivot.loc['phoneme_given_speaker','ari']:.4f}. "
          f"Speaker silhouette changes from {pivot.loc['speaker_pooled','truth_silhouette']:.4f} pooled to "
          f"{pivot.loc['speaker_given_phoneme','truth_silhouette']:.4f} within vowel; k-means ARI changes from "
          f"{pivot.loc['speaker_pooled','ari']:.4f} to {pivot.loc['speaker_given_phoneme','ari']:.4f}. "
          'These comparisons characterize this feature space; they are not a ceiling on supervised learning.')
    tab('At a glance: the four requested comparisons',formatted(pivot,['truth_silhouette','ari','nmi','ami']))
    prose('Interpretation of the results',
          f"For pooled vowels, {100*pivot.loc['phoneme_pooled','truth_negative_fraction']:.1f}% of sampled tokens have negative vowel silhouette. "
          f"In within-speaker analysis, {int((local.truth_silhouette>pivot.loc['phoneme_pooled','truth_silhouette']).sum())}/98 "
          'speakers have a higher vowel silhouette than the pooled reference. Considerable overlap remains, and speaker-to-speaker variation is substantial. '
          'For speaker identity, compare the metrics separately: conditioning can improve partition agreement or centroid separation without improving silhouette. '
          'These measures capture different aspects of unequal, overlapping distributions. '
          'Silhouette compares each token with the nearest competing category in mean distance; centroid separation and partition agreement need not improve that comparison. '
          'The data therefore contain speaker-associated structure without forming compact, well-separated speaker clouds. '
          'The more-negative-than-shuffled speaker silhouette does not mean speaker information is absent: random labels distribute every group broadly over the same space, '
          'making within- and between-group distances similar. This reference is descriptive, not a supervised-learnability test.')
    tab('Ground-truth category geometry',formatted(pivot,['truth_silhouette','shuffled_silhouette','silhouette_above_shuffle',
                                                        'truth_negative_fraction','truth_davies_bouldin','truth_between_fraction','truth_calinski_harabasz']))
    prose('Metric definitions',
          'Silhouette ranges from −1 to 1; higher means closer to the own group than the nearest competing group in mean Euclidean distance. '
          'Negative fraction counts queries with negative silhouette. Davies–Bouldin (DB) compares within-group spread to centroid separation; lower is better. '
          'Calinski–Harabasz (CH) is a degrees-of-freedom-adjusted between/within variance ratio; higher is better, but its scale depends on sample size and category count. '
          'Between fraction is between-category sum of squares divided by total sum of squares. '
          'ARI measures chance-adjusted partition agreement. NMI measures normalized shared information; AMI adjusts mutual information for chance. '
          'These are label-permutation invariant, not classification accuracies. Purity is descriptive majority-label concentration inside fitted clusters, not held-out accuracy.')
    prose('Metric reference','Definitions and limitations: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation')
    plot('ground_truth')
    tab('Unsupervised k-means versus ground truth',formatted(pivot,['ari','nmi','ami','homogeneity','completeness','purity']))
    tab('Geometry of the fitted clusters (not ground-truth categories)',formatted(pivot,['kmeans_silhouette','kmeans_davies_bouldin','kmeans_calinski_harabasz']))
    prose('Internal geometry versus categorical agreement',
          'K-means seeks compact Euclidean clusters. Its clusters can have a positive silhouette while the actual vowel or speaker categories overlap. '
          'ARI/NMI/AMI test whether these compact clusters correspond to the requested identities. '
          'K is fixed to the target category count (5 vowels or 98 speakers), not selected for favorable label agreement; labels themselves do not enter fitting. '
          'The label-free geometry is therefore conditional on this specified K, not a search for a natural number of clusters. '
          'No single composite “wellness” score is constructed, because compactness, label agreement and category size measure different properties.')
    plot('cluster_agreement')
    tab('Token-weighted sensitivity summary',formatted(weighted,['truth_silhouette','truth_between_fraction','ari','nmi','ami']))
    spread = summary[(summary.weighting=='macro')&summary.metric.isin(['truth_silhouette','ari','nmi','ami'])].copy()
    spread['view']=spread.view.map(LABELS)
    spread['metric']=spread.metric.map(DISPLAY)
    tab('Variation: repeats versus different speakers/vowels',spread[['view','metric','mean','repeat_sd','between_group_sd','minimum_group_mean','maximum_group_mean']].round(4).rename(columns=DISPLAY))
    prose('Uncertainty and conditional averages',
          'Repeat SD uses the three initialization/query seeds. For ground-truth silhouette it measures query-sampling variation; '
          'for cluster agreement it measures initialization variation on the same fit subset. '
          'Between-group SD is the heterogeneity of the 98 speaker means or five vowel means, after averaging repeats. '
          'Neither is a confidence interval for a new speaker population. '
          'Shuffled-label silhouettes preserve category sizes and share query tokens; three permutations are a descriptive reference, not a permutation significance test. '
          'Five versus 98 categories can have different random-label silhouettes; direct comparisons across targets require caution. '
          'CH changes strongly with group sample size, so its pooled-versus-conditional numerical change is not an effect size. '
          'Conditioning also changes label proportions, not only the nuisance variation.')
    tab('Speaker separation within each vowel',local_p[['group','n','truth_silhouette','truth_negative_fraction','truth_between_fraction','ari','nmi','ami']].round(4).rename(columns=DISPLAY))
    plot('phoneme_variation'); plot('speaker_variation')
    phon_counts = pd.DataFrame({'vowel':quality['phoneme_counts'].keys(),'tokens':quality['phoneme_counts'].values()})
    phon_counts['fraction'] = phon_counts.tokens/quality['n_tokens']
    tab('Category coverage',phon_counts.round(4))
    miss = pd.DataFrame({'feature':cfg['features'],'invalid_or_missing':[quality['missing_counts'][f] for f in cfg['features']]})
    miss['fraction'] = miss.invalid_or_missing/quality['n_tokens']
    tab('Missingness and cleaning',miss.round(5))
    if 'f0_median_hz' in cfg['features']:
        by_vowel = pd.read_csv(run/'missing_fraction_by_label.csv')[['label','f0_median_hz']]
        by_vowel['f0_median_hz'] = (100*by_vowel.f0_median_hz).round(2)
        tab('F0 missingness differs by vowel',by_vowel.rename(columns={'label':'Vowel','f0_median_hz':'Missing/invalid F0 (%)'}))
        f0_note = (f"F0 is missing/invalid for {100*quality['missing_counts']['f0_median_hz']/quality['n_tokens']:.2f}% of tokens. "
                   'Median-imputed F0 creates a point mass, affecting distances. ')
    else:
        f0_note = ('F0 is excluded from feature extraction, missingness handling, row selection and distance calculations. '
                   'Tokens are retained regardless of whether F0 is present; no F0 missingness indicator is used. ')
    prose('Data-quality findings',
          f"Every speaker has every vowel; no absent speaker–vowel cells. Speakers have {quality['speaker_count_min']:,}–{quality['speaker_count_max']:,} tokens. "
          f"There are {quality['duplicate_token_identifiers']} duplicated speaker/sentence/segment identifiers and "
          f"{quality['all_features_missing_rows']} all-feature-missing rows. Those rows are retained and fully imputed, not silently excluded. " +
          f0_note +
          'Nonpositive and nonfinite values are marked missing; each feature is median-imputed, then z-scored using the full dataset. '
          'No log transform, outlier clipping, speaker normalization or per-subgroup scaling is performed. '
          'Missingness by speaker and vowel is saved; no missingness indicator is included as a feature.')
    plot('pca')
    prose('Projection caveat',f"PCA is diagnostic only. All quantitative metrics use all {len(cfg['features'])} standardized dimensions. "
          'The right panel shows speaker means, whose apparent separation does not imply separated token distributions.')
    prose('Computation and reproducibility',
          f"K-means: seeds {cfg['seeds']}, {cfg['kmeans_n_init']} k-means++ restarts per seed, Lloyd algorithm, max {cfg['kmeans_max_iter']} iterations, "
          f"tolerance {cfg['kmeans_tol']}. Each group uses all tokens for assignment/evaluation and a fixed random fit subset of up to {cfg['fit_sample_max']:,} "
          f"tokens (base seed {cfg['fit_sample_seed']}, deterministic group-specific offset); smaller groups use all tokens. "
          f"Silhouette uses {cfg['query_large']:,} uniform queries per repeat for groups above {cfg['large_group_threshold']:,} tokens, "
          f"otherwise {cfg['query_small']}; each query is compared with every reference token in that group. Self distances are excluded correctly. "
          'CH, DB, variance fractions and ARI/NMI/AMI use all group tokens, not the silhouette sample. '
          'Local metrics are computed before averaging. Models, assignments, fit/query source-row IDs, preprocessing, source hash, configuration and environment are retained. '
          f"{int(frame.fit_hit_max_iter.sum())}/{len(frame)} fits reached the iteration cap; reaching the cap is flagged, not silently treated as converged. "
          'Custom query silhouettes were checked against scikit-learn on ordinary, duplicate-point and singleton-category examples.')
    prose('Use as a learning-trajectory baseline',
          'This is an in-sample descriptive snapshot, not held-out predictive performance. For future learned representations, '
          'reuse token identities, group definitions, query rows, fit subsets, seeds and aggregation rules. Define the latent-space distance/scaling rule in advance. '
          'For predictive experiments, refit preprocessing on training data only; do not reuse these all-data statistics as a train-only transformer. '
          'Tokens from the same utterance/speaker are dependent; raw token counts are not independent biological replicates. '
          'No neural model or learning trajectory was trained in this first step.')
    # Tables and full appendix remain machine-readable; HTML is self-contained.
    main_md=[]; main_html=[]
    for kind,title,value in chunks:
        if kind=='image':
            main_md.append(f'![{title}](figures/{title}.png)')
            data_url=base64.b64encode((figures/(title+'.png')).read_bytes()).decode()
            main_html.append(f'<figure><img alt="{title}" src="data:image/png;base64,{data_url}"></figure>')
        elif kind=='table':
            main_md.append('## '+title+'\n\n'+md_table(value))
            main_html.append('<h2>'+html.escape(title)+'</h2><div class="table-wrap">'+value.to_html(index=False,border=0,na_rep='—')+'</div>')
        else:
            main_md.append('## '+title+'\n\n'+value)
            main_html.append('<h2>'+html.escape(title)+'</h2><p>'+html.escape(value)+'</p>')
    detail=local[['group','n','truth_silhouette','truth_negative_fraction','truth_between_fraction','ari','nmi','ami']].round(4)
    detail.to_csv(run/'per_speaker_phoneme_summary.csv',index=False)
    main_md.append('## Per-speaker detail\n\nAll 98 rows are in per_speaker_phoneme_summary.csv and the expandable HTML appendix.')
    main_html.append('<details><summary>Full per-speaker phoneme results (98 speakers)</summary>'+detail.rename(columns=DISPLAY).to_html(index=False,border=0)+'</details>')
    (run/'REPORT.md').write_text('\n\n'.join(main_md)+'\n',encoding='utf-8')
    css='body{font:16px/1.6 Segoe UI,Arial,sans-serif;color:#203040;max-width:1180px;margin:40px auto;padding:0 24px}h2{font-size:23px;margin-top:32px;color:#193f63}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid #dce3eb}th{background:#edf2f7}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafc}.table-wrap{overflow-x:auto}img{max-width:100%;height:auto}figure{margin:25px 0}summary{cursor:pointer;font-weight:600;padding:16px 0}p{max-width:1050px} @media print{body{margin:0;font-size:11px}figure,table{break-inside:avoid}details{display:none}}'
    (run/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Vowel acoustic geometry baseline</title><style>'+css+'</style><body>'+''.join(main_html)+'</body></html>',encoding='utf-8')
    print('REPORT:',run/'REPORT.html',flush=True)
    return run/'REPORT.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');make_report(p.parse_args().run_directory)
