# Neural-only grid results

Completed run: `outputs/neural_grid/20260924_225129_ccc857`.
Executed notebook: `src/search_neural_grid.ipynb`. Environment: Anaconda `lffl`.

The original two-category statistical model, objective, latent dimension, priors,
likelihood variance, interaction penalty and initialization algorithm were retained.
The same 10 speakers, all 46,083 rows, training-only preprocessing, and 80/10/10
split were verified against the original baseline artifacts.

## Design

24 Cartesian configurations plus the original control: hidden widths/depths
`(64,)`, `(128,128)`, `(128,128,128)`; ReLU or tanh; learning rate 0.0003 or
0.001; batch size 256 or 1024. The control uses its original batch size 512.
All receive 20 pretraining epochs. Screening uses up to 30 training epochs;
the best two alternatives and control are retrained from scratch for up to 100
epochs at seeds 42, 43, 44 (34 runs total).

Individual checkpoints minimize unsupervised validation loss. Configuration
selection maximizes mean validation geometric phoneme/speaker accuracy. Thus
training gradients remain unsupervised, but hyperparameter selection uses labels.
The winner was saved before test evaluation; discarded configurations were not
ranked on test accuracy.

## Results

Validation selected c022: three 128-unit hidden layers, tanh, learning rate
0.0003, batch size 1024. Values below are test mean +/- sample SD across seeds,
in percent (SD measures seed variation, not a confidence interval).

| Configuration | Phoneme | Speaker | Both correct |
| --- | ---: | ---: | ---: |
| Original control | 39.73 +/- 14.69 | 18.11 +/- 3.05 | 7.25 +/- 4.60 |
| Selected c022 | 49.20 +/- 5.90 | 19.60 +/- 0.95 | 10.20 +/- 1.55 |

Mean gains: +9.47 percentage points phoneme, +1.49 speaker, +2.95 joint.
At seed 42, however, c022 is slightly worse on both tasks: 55.87%/20.22%
versus control 56.67%/21.59%. Gains occur at seeds 43 and 44, suggesting better
stability in this small comparison, not a uniformly better optimum.

The selected model still does not exceed the historical original single run
(56.50% phoneme, 21.39% speaker), and remains far below historical 15-NN
(82.58%, 49.79%) and classifier (83.94%, 49.19%) results. Those historical
numbers are single runs, not seed-averaged matched comparisons. The historical
control used four CPU threads; all current candidates use one thread each.

This search shows a useful average phoneme gain, but no established statistically
significant gain and little speaker improvement. Three seeds, validation-informed
selection, a previously inspected test set, and a 30-epoch screening budget limit
the conclusion. A future confirmation should freeze the selected configuration,
add prespecified seeds, and use a fresh untouched holdout before making stronger
claims. The current token split is not an unseen-speaker evaluation.

## Reproduce

```bash
python src/train_grid.py --config src/configs/neural_grid.py
python src/train.py --config src/configs/neural_grid_winner.py
```

In the notebook, the saved run is loaded by default; set `RESUME_RUN = None` for
a new search. Do not change the grid while resuming an existing run. All 22 unit
tests passed; the notebook executed end-to-end and a reloaded winning checkpoint
exactly reproduced all saved test category predictions. Previous model classes
and notebooks remain available.
