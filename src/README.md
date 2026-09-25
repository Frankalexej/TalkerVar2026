# Vowel feature learning

Start with **`train_vowels.ipynb`**. It explains and runs data preparation,
the two-factor VaDE, supervised baselines, evaluation, plots, and checkpoint reload.
The notebook and CLI call the same Python implementation.

```text
src/
  configs/vowels.py       editable full-run configuration
  configs/smoke.py        two-epoch pipeline check
  config.py              typed settings and Python config loader
  data.py                speaker selection, split, imputation, scaling
  models.py              two-factor VaDE and matched-encoder classifier
  training.py            initialization, Adam, validation, early stopping
  evaluation.py          predictions, category alignment, metrics
  experiment.py          experiment orchestration and saved artifacts
  plotting.py            learning curves and confusion matrices
  train.py               CLI entry point
  train_vowels.ipynb      complete executable walkthrough
  execute_notebook.py    execute with the current Anaconda interpreter
  tests/                 data integrity and numerical checks
```

## Run with Anaconda

On this machine the tested environment is `lffl`:

```powershell
conda activate lffl
python src/train.py --config src/configs/vowels.py
```

If `conda` is not on PATH, use an Anaconda Prompt or the environment interpreter:

```powershell
& C:\Users\46021\.conda\envs\lffl\python.exe src/train.py --config src/configs/vowels.py
```

Select that interpreter as the notebook kernel in your editor. For a new machine:

```bash
conda env create -f src/environment.yml
conda activate talkervar
python src/train.py --config src/configs/vowels.py
```

`requirements.txt` also lists dependencies. CUDA users should install a PyTorch
build compatible with their server and set `CONFIG.training.device = "cuda"`.
Every run saves exact package versions and interpreter information.

```bash
python -m unittest discover -s src/tests -v
python src/execute_notebook.py src/train_vowels.ipynb
```

The notebook runner uses the current Python executable, regardless of any globally
registered kernel. It saves cell outputs in place. Training itself always creates
a fresh timestamped result directory under `outputs/vowels/`.

## Dataset and comparison

- Five inputs: `f0_median_hz`, `f1_hz`, `f2_hz`, `f3_hz`, `duration_s`.
- Randomly select 10 speakers with seed 2026, retaining every selected token.
- Joint speaker/phoneme stratification; approximately 80% train, 10% validation,
  10% test. This is a **token split with known speakers**, not a held-out-speaker
  evaluation or a split grouped by recording. Related tokens can cross splits.
- Missing/nonfinite/nonpositive acoustic values are imputed using training-only
  medians, then scaled using training-only means and standard deviations.
- GMVAE: nonlinear Gaussian encoder/decoder, 5 x 10 diagonal Gaussian components,
  additive effects plus double-centered interaction, VaDE-style negative ELBO
  plus interaction mean-square penalty. No MI reward or label supervision.
- 15-NN: supervised uniform-vote classification in standardized acoustic space.
- Classifier: the same encoder widths and continuous bottleneck width, with two
  supervised cross-entropy heads. No decoder, so parameter counts differ.

GMVAE labels are used for category cardinalities, split stratification, and
evaluation, never its parameter updates or early stopping. Hungarian label
mapping is calibrated on the training split and frozen for held-out evaluation.
Split-optimal permutation accuracy is separately saved as a descriptive statistic.
All methods use identical rows, inputs, and preprocessing.

## Reconstruction choices

The recovered *Build Toy GMVAE* conversation specifies `(c_phoneme, c_speaker) -> z -> x`,
mixture-derived responsibilities, and ELBO plus an interaction penalty. Its last
experiment excluded MI rewards. The original real-vowel run implementation and
hyperparameters were unavailable. The configurations explicitly set new defaults:
8 latent dimensions, hidden widths 128/128, median f0, raw-value standardization,
fixed decoder variance 0.1, 20 autoencoder pretraining epochs, unlabeled sequential
residual KMeans initialization, 100 training epochs, Adam at 1e-3, and patience 20.
These are a reconstruction, not a claim to reproduce the previous run exactly.

The loss follows the recovered mean-field VaDE-style analytic Gaussian KL
description; responsibilities are Monte Carlo averages of `p(categories | z)`.
It is not the alternative collapsed structured-posterior mixture ELBO.
`interaction_weight` multiplies **mean squared** double-centered interaction,
whose normalization matters when comparing with another implementation.

## Outputs

Each fresh run records config, CSV hash, selected speakers, all source-row split
assignments, preprocessing, environment versions, per-epoch histories, validation
selected checkpoints, held-out predictions, confusion matrices, recognition
accuracies, and plots. GMVAE reports posterior/category-use diagnostics as well.
The test set does not choose epochs or label mappings. The notebook shows how to
reload checkpoints and verify predictions. Run multiple explicit seeds later if
you need uncertainty estimates; the initial full run is one seed.

The CSV remains at the repository root and is not bundled with source code.
Relative config paths resolve against the repository root, which allows the same
entry point to be called from a server/Slurm working directory later.

## Shared continuous w extension

Open **`train_shared_w.ipynb`** for the paper-inspired two-category extension.
`SharedWGMVAE` is an additional class in `models.py`; the original model classes,
baseline config, and baseline notebook remain available unchanged.

```bash
python src/train.py --config src/configs/shared_w.py
```

It adds one shared vector-valued auxiliary variable `w ~ N(0,I)` per token.
Both category axes condition the same acoustic latent `z`, and a neural network
maps the same `w` sample to means and variances for every phoneme/speaker pair.
The encoder gains mean/logvariance heads for `q(w|x)`, and the objective gains its
analytic KL to the standard normal. The decoder still receives only `z`.
The default `w_dim=2` denotes one 2D random vector; set it to 1 for a scalar.

The reference is Dilokthanakul et al., *Deep Unsupervised Clustering with Gaussian
Mixture Variational Autoencoders*, arXiv:1611.02648v2, sections 3.1–3.4 and Appendix A.
Our observed `x` is their `y`; our continuous `z` is their `x`; our categorical
pair replaces their single categorical `z`. Their auxiliary `w` keeps its name.
This extension retains the two-factor mean decomposition and penalizes the total
double-centered interaction at each sampled `w`, including the conditional output.

`SharedWModelConfig.objective` makes an estimator distinction explicit:

- `paper` (default): the paper's Eq. (5) component-KL estimator plus reconstruction,
  sample-wise categorical KL, `w` KL, and the existing interaction penalty.
- `structured`: an exact Monte Carlo ELBO estimator for the stated posterior
  `q(z|x) q(w|x) p(categories|z,w)`, retaining responsibility-dependent log-density
  terms inside the sample expectation. Tests verify its equivalence to marginalizing
  the mixture with `logsumexp`.

The paper estimator is not generally the exact structured ELBO, because its
responsibilities depend on the same `z` being integrated. The notebook derives this
distinction. It also notes the previous model's different categorical-KL averaging;
this first comparison is not a perfectly isolated causal ablation of `w` alone.
No minimum-information threshold, MI reward, or additional supervision is enabled.

New results go to `outputs/shared_w/` and include `w`-usage diagnostics. The new
notebook verifies identical CSV hash, preprocessing, and split assignments against
the saved baseline before plotting a comparison. If baseline artifacts are absent
on another machine, it can still train and evaluate the new model on its own.

## Original-model neural grid search

Use `search_neural_grid.ipynb` and `configs/neural_grid.py` to search only the
neural and optimizer settings of the original VaDE. `NeuralVaDE` adds configurable
hidden activations while inheriting the original statistical methods unchanged.

```bash
python src/train_grid.py --config src/configs/neural_grid.py
python src/train_grid.py --config src/configs/neural_grid.py --resume outputs/neural_grid/RUN_DIRECTORY
```

The default Cartesian grid is 3 width/depth settings x 2 activations x 2 learning
rates x 2 batch sizes (24 configurations), plus the original control. Every model
uses all selected training tokens. Screen for 30 epochs, select two non-control
finalists using validation geometric phoneme/speaker accuracy, then retrain those
and the control for up to 100 epochs at three seeds. Each individual checkpoint
still minimizes unsupervised validation loss. Configuration selection uses mean
validation recognition over seeds; it is explicitly label-informed selection.

No changes to latent dimension, category counts, likelihood variance, interaction
penalty, mixture priors, original loss formula, or initialization method are allowed
as grid axes. Search workers receive only train/validation arrays. Test recognition
is computed after the winning configuration is saved, only for the winner and
control across all seeds. The old test set has already been inspected, so this is
exploratory and three seeds do not establish formal statistical significance.

Every trial saves a log, history, checkpoint and validation metrics. Search output
includes screening/confirmation leaderboards, a selection record, test results,
paired seed differences, source hashes and `best_config.py`. The latter can be run
directly with `python src/train.py --config .../best_config.py`. Resume validates
the plan, source hashes and CSV hash before reusing completed trials.
