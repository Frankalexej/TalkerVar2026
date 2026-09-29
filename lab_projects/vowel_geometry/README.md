# Vowel geometry baseline (separate lab project)

This folder is independent of the GMVAE/VQ training project. The source CSV is read without modification. All 98 available speakers are included.

```powershell
& C:\Users\46021\.conda\envs\lffl\python.exe lab_projects/vowel_geometry/analyze.py --config lab_projects/vowel_geometry/config.py
& C:\Users\46021\.conda\envs\lffl\python.exe lab_projects/vowel_geometry/report.py lab_projects/vowel_geometry/results/20260929_173507
```

The executed `baseline.ipynb` provides the same workflow. It defaults to reloading the saved baseline; set `RESUME_RUN=None` for a new analysis. Existing runs require matching source/config/data hashes. A different feature set or sampling design belongs in a new run.

Four views: pooled vowels; vowels separately within each speaker; pooled speakers; speakers separately within each vowel. Conditional headline statistics are equal-weight subgroup averages, with token-weighted sensitivity summaries. ARI/NMI are calculated independently within each subgroup, never by concatenating local cluster IDs.

This is descriptive full-dataset geometry, not held-out prediction. Full-dataset median imputation and standardization intentionally define this baseline distance space. Do not use the resulting transformer as a train-only preprocessing object in future prediction experiments. F0 means **median F0**, consistent with previous feature extraction. Nonfinite/nonpositive features are imputed; no logs, clipping or subgroup rescaling is added.

The reusable `query_silhouettes` function calculates exact per-query silhouettes against complete reference groups, then uniformly sampled queries estimate each group's average. K-means fits up to 50,000 fixed sampled tokens per group and assigns all tokens. Its K equals the target category count; labels themselves are not fit inputs. Three optimization/query seeds, each with ten K-means restarts, are retained.

`REPORT.html` is a self-contained readable report, with all charts embedded and an expandable 98-speaker appendix. `REPORT.md` has the same main content. `summary.csv` separates repeat variation from subgroup variation. `group_summary.csv` and per-group folders retain all numerical details, fitted models, source-row IDs, assignments and query silhouette values. Original measurements are kept intact, with processing and results separate, following the scientific-research data-handling guidance.

The shuffled-label reference preserves category counts and uses one permutation per repeat. It is not a permutation significance test. K-means results are model-dependent, and silhouette/DB favor compact Euclidean groups; none establishes an intrinsic limit of learning. The source CSV contains token summaries, not within-vowel acoustic trajectories.

Checks: `python -m unittest lab_projects.vowel_geometry.test_metrics -v`.
# Four-feature follow-up (F0 excluded)

Use `baseline_four.ipynb` or run from the repository root:

```powershell
& 'C:\Users\46021\.conda\envs\lffl\python.exe' lab_projects/vowel_geometry/analyze.py --config lab_projects/vowel_geometry/config_four.py
```

The new configuration preserves all sampling/clustering settings, all speakers and
all rows, and selects only F1/F2/F3/duration before missing-value cleaning. F0 and
its availability have no influence. Results are saved separately in `results_four`.
`report.py RUN` generates the full report; `compare_features.py OLD_RUN NEW_RUN`
verifies identical rows, retained standardized columns, fit subsets and query rows,
then saves paired differences and a comparison report. Changing configurations
requires a new run, not resumption of a differently configured run.
