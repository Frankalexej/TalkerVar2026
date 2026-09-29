# VQ capacity, quantized speaker supervision, and raw clusterability

Completed 29 September 2026 using Anaconda `lffl`: **40 VQ runs and 120 clustering fits**, with five matched seeds (42–46). All 35 unit tests pass; the executed notebook reproduces saved checkpoint predictions exactly. Previous model definitions and results are preserved.

## Main conclusion

Larger codebooks with moderate speaker supervision improve phoneme organization substantially in this experiment. The result is not simply better reconstruction: increasing speaker supervision improves phoneme accuracy/NMI/ARI while worsening reconstruction. Factor separation improves, but remains incomplete. Raw clustering also improves after accounting for speaker differences, and the pooled GMM strongly reflects the missing-F0 imputation pattern.

The attached analysis was useful in separating discrimination, density clustering, and compression. These results support testing those mechanisms separately; they do **not** establish an intrinsic unsupervised-learning ceiling for these acoustics.

## Controlled design

- Same five features, cleaning, training-fitted imputation/standardization, and exact split as the preceding dual-VQ experiment. Train/validation/seen-test/unseen-test counts: **36,863 / 4,608 / 4,612 / 13,686**.
- Familiar speakers: jvs003, jvs009, jvs018, jvs036, jvs037, jvs038, jvs046, jvs061, jvs065, jvs078. Held-out speakers: jvs004, jvs020, jvs086. Full per-speaker counts are in the detailed report.
- Two encoders: 5→128→128→8; decoder: 16→128→128→5; ReLU hidden layers. Large codebooks: **64 phoneme codes and 64 speaker codes**, versus five vowels and ten familiar speakers. A matched **5/10-code control** separates capacity from supervision changes. Large model: 57,071 parameters.
- Speaker CE is on quantized **z_s**, replacing the earlier h_s CE; there is only one speaker loss. No phoneme supervision or adversarial objective.
- Objective: mean reconstruction MSE + both gradient-updated VQ codebook losses + both commitment losses (coefficient 0.25) + λ_s speaker CE. Standard straight-through quantization routes CE gradients into the speaker encoder/head, not directly into embedding parameters.
- λ_s = 0, 0.1, 1, 10; Adam 0.001, batch 512, 10 autoencoder-pretraining epochs, training-only K-means initialization, then **80 fixed epochs**. Final checkpoints are evaluated; no weight or checkpoint is selected using either test set. Initialization and minibatch hashes verify matching across dose conditions.
- Overcomplete-code accuracy uses a **training-fitted, frozen many-to-one majority mapping**, not one-to-one Hungarian matching. This is appropriate for multiple codes per vowel, but is not directly comparable with historical Hungarian percentages. Mapping-independent metrics are also necessary.

## Large-codebook dose response

Mean ± sample SD over five seeds. Speaker-head accuracy is defined only for familiar speakers.

| λ_s | Phoneme accuracy, seen | Phoneme accuracy, unseen | Speaker head, seen | MSE, seen / unseen |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 49.75 ± 11.87% | 47.74 ± 12.09% | Untrained | 0.0439 / 0.0438 |
| 0.1 | 61.02 ± 9.42% | 58.80 ± 9.72% | 38.40 ± 1.46% | 0.0507 / 0.0506 |
| 1 | 69.82 ± 2.02% | 67.60 ± 2.29% | 46.80 ± 0.60% | 0.0772 / 0.0788 |
| 10 | 69.18 ± 1.06% | 66.20 ± 1.72% | 45.65 ± 0.61% | 0.0880 / 0.0887 |

λ=1 is the largest observed mean, **not a newly validated optimum**. Its paired improvement over λ=0 is **20.07 ± 10.45 percentage points seen** and **19.86 ± 10.63 points unseen**. Accuracy, NMI, and ARI improve in all five matched seeds on both tests.

| Seed | Seen accuracy difference, pp | Unseen accuracy difference, pp |
| ---: | ---: | ---: |
| 42 | +31.14 | +28.57 |
| 43 | +11.47 | +10.73 |
| 44 | +16.89 | +16.46 |
| 45 | +31.16 | +33.47 |
| 46 | +9.69 | +10.07 |

The effect is not larger for unfamiliar speakers. At λ=1, phoneme NMI rises from **0.163→0.283 seen** and **0.169→0.281 unseen**; ARI rises **0.035→0.067** and **0.043→0.078**. ARI remains low partly because 64 codes partition five categories into many subdivisions; this does not make ARI irrelevant or establish perfect categorical organization.

Four-way mean NMI for λ=0→1:

| Association | Familiar speakers | Held-out speakers |
| --- | ---: | ---: |
| Phoneme codes ↔ phoneme | 0.163→0.283 | 0.169→0.281 |
| Phoneme codes ↔ speaker | 0.125→0.089 | 0.120→0.099 |
| Speaker codes ↔ phoneme | 0.203→0.110 | 0.206→0.093 |
| Speaker codes ↔ speaker | 0.144→0.303 | 0.122→0.214 |

Thus supervision moves information in the intended direction, without eliminating cross-factor information. At λ=10, phoneme-code speaker contamination rebounds to 0.114 seen / 0.121 unseen, providing no further phoneme benefit.

No catastrophic codebook collapse: at λ=1, approximately 63 phoneme and 61 speaker codes are active on seen tokens; perplexities are 52.6 and 53.3. Replacing the phoneme or speaker branch with its training-population mean increases seen MSE by 0.604 or 0.311, respectively (unseen: 0.596 / 0.283). Both branches matter, although the phoneme branch has greater reconstruction influence.

The 5/10-code control at λ=1 reaches 49.01 ± 7.49% seen / 46.97 ± 8.86% unseen using the same many-to-one metric, with MSE 0.2880 / 0.2837. Larger codebooks clearly relax the reconstruction bottleneck. Some accuracy advantage also comes from the more flexible code-to-label mapping; cross-capacity accuracy alone is not proof of better disentanglement. Comparisons with the previous h_s-supervised run are historical, because that experiment selected checkpoints by validation reconstruction.

## Raw clusterability

Five clusters/components; K-means has 10 restarts, full-covariance GMM five restarts. All fitting and label mappings use training data only. Results below are seen-test **frozen Hungarian accuracy**, mean ± SD.

| Method | Pooled | Individual-speaker models | Speaker-normalized pooled |
| --- | ---: | ---: | ---: |
| K-means | 35.42 ± 0.05% | 49.51 ± 0.69% | 43.49 ± 0.03% |
| Full-covariance GMM | 35.20 ± 0.01% | 48.46 ± 1.08% | 41.03 ± 0.04% |

Individual scores are token-weighted averages of within-speaker scores; arbitrary local cluster IDs are never pooled. These ten separate models also have more total capacity, so their gain is not a pure speaker-information effect. Normalization uses each familiar speaker's training mean/SD, frozen for evaluation. Neither individual fits nor normalization is applied to held-out speakers without training statistics. Pooled unseen accuracy is 33.52 ± 0.04% for K-means and 34.16 ± 0.01% for GMM. All 120 final fits converged.

Speaker normalization improves NMI/ARI too: K-means NMI 0.133→0.184 and ARI 0.100→0.145; GMM NMI 0.155→0.269 and ARI 0.090→0.228. Speaker variation therefore interferes with these pooled clustering solutions, but simple location/scale correction does not recover clean vowel categories.

**Important diagnostic:** pooled GMM assignments have NMI **0.518 with missing/invalid F0**, compared with 0.155 with vowel identity. Training median imputation creates a point mass for 32.6% of training tokens. The GMM is strongly associated with that artifact. After speaker normalization, missingness NMI is 0.142; normalization also transforms the global imputation value into speaker-specific values, so its gain cannot be attributed solely to removing speaker variation. This is a post-fit association, not a causal F0-ablation result.

## Artifacts and limits

- Executed notebook: [src/train_vq_sensitivity.ipynb](src/train_vq_sensitivity.ipynb).
- Configuration: [src/configs/vq_sensitivity.py](src/configs/vq_sensitivity.py); CLI: `python src/train_sensitivity.py --config src/configs/vq_sensitivity.py`.
- [Full report, all dose/capacity tables and plots](outputs/vq_sensitivity/20260929_011130_ec2936/REPORT.md).
- Run directory: `outputs/vq_sensitivity/20260929_011130_ec2936`. Contains source/config snapshots, preprocessing, split manifest, checkpoints, raw fitted models, training histories, predictions, mappings, feature-wise errors, branch ablations, contingencies, occupancy and paired per-seed CSVs.

Five optimization seeds on one speaker sample do not estimate uncertainty across speaker populations. Familiar-speaker splits are token-level, not utterance-held-out. These experiments support improved unsupervised phoneme organization from quantized speaker supervision and extra code capacity, not fully identified phoneme/speaker factors. No probes, context pooling, F0-removal experiment or additional model family was silently added.
