# VQ capacity and raw clusterability follow-up

Use Anaconda `lffl`, from the repository root:

```bash
python src/train_sensitivity.py --config src/configs/vq_sensitivity.py
python src/train_sensitivity.py --config src/configs/vq_sensitivity.py --resume outputs/vq_sensitivity/RUN_ID
python src/sensitivity_report.py outputs/vq_sensitivity/RUN_ID
```

`train_vq_sensitivity.ipynb` runs/resumes the experiment and verifies checkpoints.
Its default loads the saved run; set `RESUME_RUN=None` to train anew.

The previous models and their source files remain unchanged. New modules:

- `sensitivity_models.py`: quantized speaker CE, configurable codebook sizes.
- `sensitivity_training.py`: matched initialization, fixed 80-epoch dose responses.
- `sensitivity_metrics.py`: train-fitted many-to-one mapping, code organization,
  reconstruction and branch-ablation metrics.
- `raw_clusterability.py`: pooled, separate-speaker and normalized-speaker clustering.
- `sensitivity_experiment.py`: source/data-verified resume and experiment orchestration.
- `sensitivity_report.py`: full report, dose-response and raw-clustering figures.

Data is exactly matched to the prior dual VQ run, including all four splits and
training-only preprocessing. Raw clustering uses K=5. Per-speaker normalization
uses only familiar speakers' training tokens; no calibration or adaptation on
held-out-speaker tests is performed. Individual models use separate train-fitted
code permutations; do not pool their raw cluster IDs to compute NMI/ARI.

The VQ experiment crosses 5/10 and 64/64 codes with lambda 0, 0.1, 1, 10 at five
matched seeds. CE is on the straight-through quantized speaker representation.
All final-epoch results are reported. This is a dose-response experiment, not a
search choosing one reconstruction-optimal weight. The zero-weight head is not
trained/evaluated. No phoneme identities enter VQ training. Large-code phoneme
accuracy uses frozen train-majority mappings; these are many-to-one and should
not be confused with historical one-to-one Hungarian accuracy. NMI, ARI and AMI
supplement accuracy because code granularity can itself change these metrics.

No context pooling, representation probes, F0 changes or phoneme-label losses
are included. All logs, initial/final checkpoints, mappings, fit convergence
records, numeric metrics, paired contrasts and source snapshots are saved.
