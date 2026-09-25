# TalkerVar2026 Experiment Report

## Scope

The project reconstructed and evaluated the vowel-model experiments discussed in
the earlier GPT Work conversation. The input is `metadata_vowels_acoustics.csv`,
using median F0, F1, F2, F3, and duration. Ten speakers were selected reproducibly
with seed 2026; all 46,083 observations from those speakers were retained and split
80/10/10 into 36,866 training, 4,608 validation, and 4,609 test observations.
Missing-value imputation and standardization were fitted on training data only.

## Results

Single-run baseline test results:

| Model | Phoneme | Speaker | Joint |
| --- | ---: | ---: | ---: |
| Original GMVAE | 56.50% | 21.39% | 12.22% |
| GMVAE with shared continuous variable | 38.47% | 19.85% | 8.09% |
| 15-NN | 82.58% | 49.79% | 44.76% |
| Supervised classifier | 83.94% | 49.19% | 43.96% |

The shared-continuous-variable extension made performance worse. The neural search
selected three 128-unit tanh layers, learning rate 0.0003, and batch size 1024.
Matched three-seed test results were:

| Configuration | Phoneme, mean +/- SD | Speaker, mean +/- SD | Joint, mean +/- SD |
| --- | ---: | ---: | ---: |
| Original control | 39.73 +/- 14.69% | 18.11 +/- 3.05% | 7.25 +/- 4.60% |
| Selected configuration | 49.20 +/- 5.90% | 19.60 +/- 0.95% | 10.20 +/- 1.55% |

The selected network improved mean phoneme accuracy by 9.47 percentage points and
was more stable, but speaker accuracy improved by only 1.49 points. At seed 42 it
was slightly worse than the control, so three seeds do not establish statistical
significance. It also remains well below the kNN and supervised baselines. Because
the test set had been inspected in earlier experiments, these comparisons should be
treated as exploratory; a fresh frozen holdout and more prespecified seeds are
needed for a confirmatory result.