# Four-feature VQ-VAE / GMVAE comparison

Completed 29 September 2026 in Anaconda `lffl`: **35 neural runs, one deterministic 15-NN fit, and ten pooled clustering fits**. Five optimization seeds: 42–46. All 39 tests pass. The executed notebook exactly reproduces saved VQ-64 and GMVAE-64 held-out predictions and verifies manifest identity and matched VQ initialization/minibatches.

## Main findings

Removing F0 helps pooled density clustering, but **is not an overall improvement**. It improves the mean unsupervised VQ-64 result while weakening speaker-supervised VQ-64 and slightly weakening supervised references. Thus problematic F0 imputation and useful predictive information in F0 can coexist.

Speaker supervision still improves VQ phoneme accuracy on average, with substantial seed variation. Increasing K modestly improves many-to-one accuracy, but does not consistently improve categorical organization or factor separation. Unsupervised VQ does not clearly outperform the original GMVAE.

## Design and fidelity

- Features and reconstruction target: **F1, F2, F3, duration only**. Neither F0 nor a missingness indicator is loaded. Training-only median imputation and standardization match the corresponding four old feature statistics.
- The previous manifest is loaded directly and copied byte-for-byte. Counts: **36,863 train / 4,608 validation / 4,612 familiar-speaker test / 13,686 held-out-speaker test**. All five vowels occur in all splits.
- Familiar speakers: jvs003, jvs009, jvs018, jvs036, jvs037, jvs038, jvs046, jvs061, jvs065, jvs078. Held out: jvs004, jvs020, jvs086. Exact per-speaker/category counts are in the full report.
- VQ retains two 128×128 ReLU encoders, two 8D quantized vectors, and a 128×128 ReLU decoder. Speaker CE is on **z_s**, never h_s. K=16/16 or 64/64; λ=0 or 1. Original MSE + codebook + 0.25 commitment + λ speaker CE; ten AE-pretraining epochs, K-means initialization, Adam .001, batch512, 80 fixed epochs, final checkpoint. No phoneme supervision. Optional λ=.1/10 doses were not repeated.
- GMVAE is the **original TwoFactorVaDE**, not the shared-w extension or neural-search winner. It retains the original 8D Gaussian latent, 128×128 ReLU network, Gaussian reconstruction NLL (variance .1), Gaussian/categorical KLs and interaction penalty10. Twenty AE epochs, residual K-means initialization, Adam .001, batch512, up to100 epochs, original validation-objective stopping/checkpoint rule. No speaker or phoneme supervision.
- To accommodate K² Gaussian components, quadratic calculations use equivalent matrix products rather than large broadcast tensors. Loss/gradient equivalence tests pass, including checks on trained checkpoints. GMVAE uses deterministic CUDA on RTX3060; VQ/classifier retain CPU execution. No mixed precision or altered statistical objective.
- K16→K64 changes VQ parameters **55,918→56,686**, but GMVAE parameters **41,788→104,092**: the latter has **256→4,096 joint Gaussian components**, not 2K embeddings.
- New conditions share manifest/source-row ordering. Historical input-dimensionality changes also change initialization, and the old randomized array order was not stored in the manifest; historical same-seed comparisons are not identical-initialization interventions.

## Phoneme accuracy

Mean ± sample SD. Discrete models use **training-fitted, frozen many-to-one mappings**. References use supervised labels. No test-based configuration selection was performed.

| Model | K per factor | Speaker CE λ | Familiar speakers | Held-out speakers |
| --- | ---: | ---: | ---: | ---: |
| VQ-VAE | 16 | 0 | 55.59 ± 10.80% | 53.75 ± 11.51% |
| VQ-VAE | 16 | 1 | 61.01 ± 6.51% | 59.29 ± 6.27% |
| VQ-VAE | 64 | 0 | 58.18 ± 7.08% | 56.08 ± 6.45% |
| VQ-VAE | 64 | 1 | 65.02 ± 5.00% | 59.94 ± 6.32% |
| Original GMVAE | 16 | None | 56.58 ± 2.83% | 53.66 ± 3.10% |
| Original GMVAE | 64 | None | 59.22 ± 1.03% | 54.85 ± 1.62% |
| Supervised classifier | — | Supervised reference | 80.25 ± 0.31% | 76.71 ± 0.32% |
| 15-NN | — | Supervised reference | 80.36% | 77.06% |

## Mandatory paired effects

Differences in percentage points, mean ± sample SD of **within-seed differences**, not differences of SDs. Full per-seed values and corresponding NMI/AMI/ARI/leakage differences are saved in `paired_effects.csv`.

| Contrast | Familiar speakers | Held-out speakers |
| --- | ---: | ---: |
| VQ16: λ1 − λ0 | +5.41 ± 8.58 | +5.54 ± 8.12 |
| VQ64: λ1 − λ0 | +6.83 ± 6.76 | +3.86 ± 7.85 |
| VQ λ0: K64 − K16 | +2.59 ± 5.61 | +2.33 ± 7.42 |
| VQ λ1: K64 − K16 | +4.01 ± 3.84 | +0.66 ± 4.27 |
| GMVAE: K64 − K16 | +2.64 ± 2.55 | +1.19 ± 4.68 |
| K16: unsupervised VQ − GMVAE | −0.99 ± 12.90 | +0.09 ± 11.99 |
| K64: unsupervised VQ − GMVAE | −1.04 ± 7.92 | +1.23 ± 6.06 |

Speaker supervision does not help every seed: accuracy improves in 4/5 familiar and 3/5 held-out seed pairs at K16, and 4/5 on both tests at K64. Its benefit is not larger on unfamiliar speakers at K64. These are exploratory effects, not established population-level significance.

## Organization, factor separation, and reconstruction

At K64, supervision raises phoneme NMI **0.214→0.253 seen / 0.214→0.242 unseen**, and reduces phoneme-code speaker NMI **0.116→0.070 / 0.100→0.066**. Speaker-code speaker NMI rises **0.103→0.250 / 0.091→0.199**. However, speaker-code phoneme NMI also rises **0.148→0.188 / 0.160→0.175**: cleaner phoneme codes do not imply complete two-way disentanglement. Seen speaker-head accuracy is 41.82 ± 0.59%; it is undefined for held-out identities.

K16 with supervision has **higher phoneme NMI** (0.274 seen / 0.273 unseen), **lower speaker contamination** (0.031 / 0.046), and higher phoneme ARI (0.156 / 0.163) than K64 (0.058 / 0.060 ARI). K64's extra subdivisions make majority mapping more flexible; its small held-out accuracy advantage does not establish better categorical discovery. NMI/AMI/ARI themselves depend on partition granularity, so none should be interpreted alone.

GMVAE has higher phoneme NMI/ARI than unsupervised VQ at both capacities. At K64 its phoneme NMI is **0.268 / 0.258** and ARI **0.124 / 0.140**, but speaker-code speaker NMI is only **0.071 / 0.074**. Its intended speaker variable is poorly organized, and factor names are not guaranteed semantics under its unsupervised objective. No test-driven axis swapping was used.

There is no complete one-code collapse, but GMVAE substantially underuses K64: seen hard assignments occupy approximately **34 phoneme / 29 speaker states**, with perplexities **19.3 / 8.9**. VQ K64 λ1 occupies approximately **63 / 61 states**. More Gaussian components do not translate into comparable effective discrete capacity.

| Four-feature model | Standardized MSE, seen | Unseen |
| --- | ---: | ---: |
| VQ16 λ0 | 0.0808 ± 0.0050 | 0.0863 ± 0.0038 |
| VQ16 λ1 | 0.1181 ± 0.0073 | 0.1259 ± 0.0092 |
| VQ64 λ0 | 0.0281 ± 0.0013 | 0.0293 ± 0.0014 |
| VQ64 λ1 | 0.0491 ± 0.0020 | 0.0539 ± 0.0033 |
| GMVAE16 | 0.0224 ± 0.0012 | 0.0248 ± 0.0014 |
| GMVAE64 | 0.0221 ± 0.0011 | 0.0249 ± 0.0014 |

GMVAE reconstructs through continuous z; VQ reconstructs through quantized vectors. GMVAE's lower MSE therefore does not prove superior discrete factorization. No raw MSE comparison between four- and five-feature targets is made.

## Removing F0: historical paired comparisons

For VQ64, four-feature minus five-feature phoneme accuracy:

| Condition | Familiar speakers, pp | Held-out speakers, pp |
| --- | ---: | ---: |
| λ0 | +8.43 ± 15.58 | +8.34 ± 15.18 |
| λ1 | −4.80 ± 5.45 | −7.66 ± 7.03 |

Supervised phoneme accuracy also declines slightly: classifier **−1.43 ± 0.47 / −1.74 ± 0.25 pp** on the three common historical seeds42–44; 15-NN **−1.65 / −1.32 pp**. This suggests F0 was not merely useless noise, although historical optimization differences prevent a perfectly isolated causal estimate.

Pooled raw clustering improves substantially. The table uses frozen training-fitted **Hungarian** accuracy, unlike overcomplete-code majority accuracy above:

| Method | Four-feature seen | Four-feature unseen | Change from five features, seen / unseen |
| --- | ---: | ---: | ---: |
| K-means, K5 | 45.72 ± 0.15% | 41.16 ± 0.18% | +10.30 / +7.63 pp |
| Full-covariance GMM, K5 | 49.41 ± 0.14% | 51.18 ± 0.04% | +14.22 / +17.02 pp |

All ten fits converged. Seen GMM NMI rises from approximately 0.155 to 0.280 and ARI from 0.090 to 0.254. This is consistent with F0/imputation distorting the earlier density clusters, but removal changes both the acoustic information and the imputation artifact; it does not identify their separate contributions. No new missing-F0 analysis was run.

## Reproduction and limitations

- [Executed notebook](src/train_four_feature.ipynb)
- [Configuration](src/configs/four_feature.py) and [usage notes](src/FOUR_FEATURE_README.md)
- [Full comparison table, all metrics and plots](outputs/four_feature/20260929_115725_f5d4e5/REPORT.md)
- Run directory: `outputs/four_feature/20260929_115725_f5d4e5`. Includes source/config snapshots, environment, manifest, preprocessing, checkpoints, per-seed metrics, frozen mappings, predictions, contingencies, occupancy, training histories, feature-wise errors and paired/historical effects.

Run with `python src/train_four_feature.py --config src/configs/four_feature.py`; the notebook defaults to reloading the saved run. Previous models/results remain untouched.

These results cover one fixed speaker sample and five optimization seeds, not uncertainty across speaker populations. Familiar-speaker splits are token-level, not utterance-held-out. Both test sets have been inspected in earlier experiments. Comparing speaker-supervised VQ with unsupervised GMVAE mixes supervision with model family; even the cleaner unsupervised comparison retains the original families' different objectives, latent structures and checkpoint rules.
