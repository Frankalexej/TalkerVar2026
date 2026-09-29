# Dual-codebook VQ-VAE: experiment summary

Speaker supervision modestly improved unsupervised phoneme organization on average,
but did not produce clean or consistently speaker-independent discrete factors.
The full technical report includes all per-feature errors, NMI/ARI/purity, code
usage, paired seed differences, and diagnostic plots:
[Full report](outputs/dual_vq/20260928_190154_268b85/REPORT.md).

## Data and protocol

Training-pool speakers: **jvs003, jvs009, jvs018, jvs036, jvs037, jvs038, jvs046,
jvs061, jvs065, jvs078**. Held-out speakers: **jvs004, jvs020, jvs086**.
The original ten-speaker pool was retained; the additional three were sampled
reproducibly with seed 2026. Each familiar speaker was independently split by token.

| Split | Observations |
| --- | ---: |
| Training | 36,863 |
| Validation | 4,608 |
| Familiar-speaker test | 4,612 |
| Unfamiliar-speaker test | 13,686 |

All five vowels occur in every split. Input/target features are median F0, F1, F2,
F3 and duration. Invalid/nonpositive values are imputed using training medians;
standardization is fitted on training data only. Exact per-speaker and per-vowel
counts are in the full report. Familiar-speaker splits are not utterance-grouped.

Two separate encoders each use `5 → 128 ReLU → 128 ReLU → 8`; codebook sizes are
5 and 10. The decoder uses `16 → 128 ReLU → 128 ReLU → 5`. A linear speaker head
reads the pre-quantized speaker embedding. Total parameters: **56,167**.

The loss is mean standardized reconstruction squared error, plus codebook MSE and
0.25 times commitment MSE for each branch, plus speaker CE weighted by lambda.
Codebooks use gradient updates and a straight-through estimator. Initialization
uses ten continuous-autoencoder epochs and training-only K-means. Adam learning
rate is 0.001, batch size 512; final training lasts 80 epochs, selecting the minimum
validation reconstruction checkpoint.

Validation-only selection among lambda **0.1, 1, 10** chose **0.1**, using
reconstruction, speaker-code NMI and unused-code penalties. Phoneme identity was
excluded from VQ training and configuration selection. Condition A uses lambda
0.1; condition B uses zero, with identical architecture, initialization and
minibatch order, verified by hashes. Final matched seeds: **42–46**. Training-label
Hungarian mappings were frozen and applied to both test sets.

## Primary five-seed result

Values are phoneme accuracy, mean ± sample SD (%).

| Condition | Familiar speakers | Unfamiliar speakers |
| --- | ---: | ---: |
| A: speaker-supervised | 39.18 ± 11.13 | 39.30 ± 11.08 |
| B: unsupervised | 36.22 ± 9.79 | 35.76 ± 9.32 |
| Paired A − B, percentage points | +2.96 ± 3.97 | +3.54 ± 5.88 |

Accuracy improved in **4/5 familiar** and **3/5 unfamiliar** seed pairs. Seed 44
contributed the largest gain. Mean phoneme NMI increased by 0.049/0.059 and ARI by
0.044/0.051 (familiar/unfamiliar). Speaker contamination NMI decreased by
0.016/0.009 on average, but decreased in only **2/5 unfamiliar-speaker pairs**.
Thus the average direction is encouraging, while a reliable improvement in
speaker independence is not established.

## What the diagnostics show

- A's familiar-speaker classifier accuracy is **39.26%**, versus **21.71%** for
  its independently evaluated, train-mapped speaker codes. Continuous speaker
  classification does not translate directly into discrete speaker organization.
- Mean familiar-speaker NMI for A is: phoneme-code/phoneme **0.187**,
  phoneme-code/speaker **0.065**, speaker-code/phoneme **0.162**, and
  speaker-code/speaker **0.153**. Speaker codes still carry substantial phoneme
  information.
- Every final model uses all **5 phoneme and 10 speaker codes** on both tests.
  The weak factorization is not explained by completely unused codebooks.
- Replacing A's phoneme/speaker codes with their training-population means raises
  familiar-test MSE by **0.335/0.501**. Both branches contribute to reconstruction;
  neither is simply ignored.
- Standardized reconstruction MSE is **0.2012/0.2007** for A and
  **0.2017/0.1964** for B (familiar/unfamiliar). There is no average reconstruction
  degradation on this held-out trio, but A does not improve unseen reconstruction
  over B.

## Matched baseline comparison

The new per-speaker token split differs from the historical split. All baselines
were therefore retrained. This table uses the **same three seeds 42–44** for every
neural model; it is distinct from the five-seed table above. 15-NN is deterministic.

| Model | Familiar phoneme accuracy (%) | Unfamiliar phoneme accuracy (%) |
| --- | ---: | ---: |
| VQ-VAE A | 41.63 ± 14.21 | 42.71 ± 13.08 |
| VQ-VAE B | 38.34 ± 12.87 | 38.26 ± 12.16 |
| Original GMVAE | 38.49 ± 1.72 | 35.98 ± 4.54 |
| Shared-continuous-variable GMVAE | 37.31 ± 8.16 | 35.22 ± 8.98 |
| 15-NN | 82.00 | 78.38 |
| Supervised classifier | 81.67 ± 0.16 | 78.32 ± 0.38 |

A improves mean phoneme accuracy over the GMVAE references on these matched seeds,
but is considerably more variable than the original GMVAE. The supervised
references use phoneme labels during training and are not equivalent unsupervised
comparisons. GMVAE reconstruction benefits from continuous latent variation;
the VQ decoder can emit at most 50 distinct vectors from its hard code pairs.

The likely limitation is that speaker CE organizes the continuous speaker
representation without forcing its ten nearest-neighbor codes to represent
speaker alone. Reconstruction rewards retaining acoustic variation wherever it
helps, and neither branch excludes the other factor. The results support a modest
effect at the selected speaker weight, not a solved factorization problem.
Five optimization seeds and one held-out speaker trio also limit broader inference.

## Reproduce and inspect

- [Executed notebook](src/train_dual_vq.ipynb)
- [Configuration](src/configs/dual_vq.py)
- [Implementation guide](src/DUAL_VQ_README.md)
- [Full results and diagnostics](outputs/dual_vq/20260928_190154_268b85/REPORT.md)
- [Paired per-seed differences](outputs/dual_vq/20260928_190154_268b85/paired_differences.csv)

Run `python src/train_vq.py --config src/configs/dual_vq.py` in Anaconda `lffl`.
The notebook loads the completed run by default; set `RESUME_RUN=None` for a fresh
experiment. Checkpoints, mappings, preprocessing, source hashes, environment
versions, metrics and figures are saved in the run directory. All 30 unit tests
passed; notebook execution additionally checks exact checkpoint reproduction.
