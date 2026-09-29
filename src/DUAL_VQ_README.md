# Dual-codebook VQ-VAE

Run from the repository root with Anaconda `lffl`:

```powershell
& C:\Users\46021\.conda\envs\lffl\python.exe src/train_vq.py --config src/configs/dual_vq.py
```

The same entry point works on a server with `python src/train_vq.py --config
src/configs/dual_vq.py`. `src/train_dual_vq.ipynb` runs/resumes the full workflow,
displays results, generates a report and diagnostics, and checks saved predictions.
The notebook defaults to the saved experiment; set `RESUME_RUN=None` for a new run.

- `vq_data.py`: retained ten-speaker pool, three reproducibly held-out speakers,
  independent 80/10/10 splits, training-only imputation and standardization.
- `vq_models.py`: two nonlinear encoders, gradient-updated VQ codebooks, nonlinear
  decoder, pre-quantization speaker classifier.
- `vq_training.py`: acoustic-only continuous pretraining/K-means initialization,
  supervised or unsupervised VQ training, fixed budgets and validation checkpointing.
- `vq_metrics.py`: train-fitted frozen Hungarian mapping, code information/usage,
  reconstruction and branch-removal diagnostics on both tests.
- `vq_baselines.py`: original GMVAE, shared-w GMVAE, classifier and 15-NN reruns.
- `vq_experiment.py`: selection and final stages, resumable saved trials, seed
  pairing and audit hashes, test evaluation only after selection and training.
- `vq_report.py`: all diagnostic plots and `REPORT.md`.

For a stopped run, add `--resume outputs/dual_vq/RUN_ID`. Completed trials are
reused only under matching plan/source/data hashes. Each run snapshots training
code and records environment versions. Reports can be regenerated independently:

```bash
python src/vq_report.py outputs/dual_vq/RUN_ID
python -m unittest src.tests.test_dual_vq -v
```

Protocol: tune speaker CE weight (0.1, 1, 10) on validation reconstruction, speaker
code NMI and active-code fractions at three seeds, then compare A (chosen weight)
and B (zero) at five matched seeds. Phoneme labels never enter VQ gradient training
or configuration selection. A/B training caches contain acoustics and speaker IDs
only. The supervised reference baselines are explicitly allowed phoneme labels.

The original speaker pool is preserved, but the requested independently random
within-speaker splits differ from the previous joint speaker×phoneme stratification.
All baselines are rerun on the new split. Historical numbers are not used as matched
comparisons. Unseen identities are evaluated by code structure, not by accuracy of
the familiar-speaker classifier. B's inactive classifier head is not evaluated.
