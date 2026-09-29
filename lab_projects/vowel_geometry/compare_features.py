"""Verify matched designs and report four-minus-five feature geometry changes."""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from lab_projects.vowel_geometry.report import VIEWS, LABELS, MEASURES, aggregate
from src.vq_report import md_table


def compare(old, new):
    old, new = Path(old), Path(new)
    plans = [json.loads((r/'plan.json').read_text()) for r in (old, new)]
    assert plans[0]['csv_sha256'] == plans[1]['csv_sha256'], 'Dataset changed'
    configs = [p['config'] for p in plans]
    assert configs[0]['features'] == ['f0_median_hz','f1_hz','f2_hz','f3_hz','duration_s']
    assert configs[1]['features'] == configs[0]['features'][1:]
    for key in configs[0]:
        if key not in ('features', 'output_dir'):
            assert configs[0][key] == configs[1][key], key
    for name in ('source_rows.npy', 'phoneme.npy', 'speaker.npy'):
        np.testing.assert_array_equal(np.load(old/name), np.load(new/name))
    np.testing.assert_allclose(np.load(old/'standardized.npy')[:,1:],
                               np.load(new/'standardized.npy'), rtol=0, atol=1e-12)
    checked = 0
    for group in sorted((old/'groups').iterdir()):
        counterpart = new/'groups'/group.name
        np.testing.assert_array_equal(np.load(group/'fit_source_rows.npy'),
                                      np.load(counterpart/'fit_source_rows.npy'))
        for seed in configs[0]['seeds']:
            filename = f'silhouette_queries_seed_{seed}.csv'
            # Query identities and truth labels match, but fitted clusters need not.
            a, b = [pd.read_csv(r/filename) for r in (group, counterpart)]
            pd.testing.assert_frame_equal(a[['source_row','truth']], b[['source_row','truth']])
            checked += 1
    frames = [pd.read_csv(r/'metrics.csv', dtype={'group':str}) for r in (old,new)]
    keys = ['view','group','seed']
    a,b = [f.set_index(keys).sort_index() for f in frames]
    assert a.index.equals(b.index)
    paired = b[MEASURES] - a[MEASURES]
    paired.to_csv(new/'paired_feature_differences.csv')
    rows = []
    for view in VIEWS:
        # Equal-weight local group average within each matched repeat.
        for metric in MEASURES:
            delta = paired.xs(view,level='view')[metric].groupby('seed').mean()
            rows.append({'view':view,'metric':metric,
                         'five_features':a.xs(view,level='view')[metric].mean(),
                         'four_features':b.xs(view,level='view')[metric].mean(),
                         'delta_four_minus_five':delta.mean(), 'paired_repeat_sd':delta.std(ddof=1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(new/'feature_comparison.csv',index=False)
    paragraphs = [
        '# Four-feature versus five-feature baseline',
        'All 447,265 tokens, 98 speakers and five vowels; F1, F2, F3 and duration only in the new run. '
        'F0 availability never filters tokens and no F0 missingness indicator is used. '
        'The 162 rows missing all four retained features remain median-imputed, as in the original protocol.',
        f'Checks passed: identical token IDs, labels, the four standardized columns, all 105 fitting subsets and all {checked} query subsets. '
        'K-means seeds 42, 43 and 44 and all other settings are unchanged. '
        'Removing a dimension changes Euclidean geometry and can change k-means++ initial centers despite matched random seeds.',
        'Conditional results use equal-weight subgroup means (98 speakers or five vowels). '
        'ARI/NMI/AMI compare unsupervised k-means assignments with ground truth; silhouette here uses ground-truth categories. '
        'All values describe this dataset, not held-out classification. Both runs fit imputation and standardization on all observations.',
    ]
    paragraphs.append('## Main findings')
    for view in VIEWS:
        part = summary[summary.view.eq(view)].set_index('metric')
        values = []
        for metric in ['truth_silhouette','ari','nmi']:
            row = part.loc[metric]
            values.append(f"{metric}: {row.five_features:.4f} → {row.four_features:.4f} (change {row.delta_four_minus_five:+.4f})")
        paragraphs.append(LABELS[view]+'. '+'; '.join(values)+'.')
    for metric in ['truth_silhouette','ari','nmi','ami','truth_davies_bouldin','truth_between_fraction','kmeans_silhouette']:
        table = summary[summary.metric.eq(metric)].drop(columns='metric').copy()
        table['view'] = table.view.map(LABELS)
        paragraphs += ['## '+metric, md_table(table.round(5))]
    paragraphs += [
        '## Interpretation and limitations',
        'Read positive four-minus-five differences as improvement for silhouette, ARI, NMI, AMI and between-category variance; '
        'DB has the opposite direction. These metrics need not agree. The paired repeat SD is not a population confidence interval: '
        'it measures k-means initialization or silhouette query variation, not independent speaker sampling.',
        'Excluding F0 removes its numerical and imputation-related contribution to distance. It does not remove all pitch-correlated '
        'or speaker-related information from the remaining acoustics, nor establish that missingness in those remaining features is harmless. '
        'This comparison cannot separate the loss of useful F0 information from the removal of F0 imputation artifacts. '
        'It is a feature-ablation baseline, not a causal estimate or a ceiling on nonlinear supervised learning.',
        'Full four-feature results and plots: [REPORT.html](REPORT.html). '
        'Exact group/seed differences: paired_feature_differences.csv. All metric comparisons: feature_comparison.csv.',
        f'Original run: {old.resolve()}\n\nFour-feature run: {new.resolve()}',
    ]
    (new/'COMPARISON.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    pieces = []
    for paragraph in paragraphs:
        if paragraph.startswith('|'):
            # The authoritative tables come directly from the numeric summary.
            continue
        elif paragraph.startswith('## '):
            metric = paragraph[3:]
            pieces.append('<h2>'+html.escape(metric)+'</h2>')
            if metric in MEASURES:
                t = summary[summary.metric.eq(metric)].drop(columns='metric').copy()
                t['view'] = t.view.map(LABELS)
                pieces.append(t.round(5).to_html(index=False,border=0))
        elif paragraph.startswith('# '):
            pieces.append('<h1>'+html.escape(paragraph[2:])+'</h1>')
        else:
            pieces.append('<p>'+html.escape(paragraph)+'</p>')
    pieces.append('<p><a href="REPORT.html">Full report and diagnostic plots</a></p>')
    css = 'body{font:16px/1.6 Segoe UI,Arial;max-width:1120px;margin:40px auto;padding:20px;color:#203040}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f7}td:first-child{text-align:left}'
    (new/'COMPARISON.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Four versus five features</title><style>'+css+'</style><body>'+''.join(pieces)+'</body></html>',encoding='utf-8')
    (new/'comparison_checks.json').write_text(json.dumps({'fit_subsets_checked':105,'query_subsets_checked':checked,
        'standardized_retained_columns_match':True,'row_ids_and_labels_match':True,
        'old_run':str(old.resolve()),'new_run':str(new.resolve())},indent=2),encoding='utf-8')
    print(summary[summary.metric.isin(['truth_silhouette','ari','nmi','ami'])].to_string(index=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old_run'); parser.add_argument('new_run')
    args = parser.parse_args()
    compare(args.old_run,args.new_run)
