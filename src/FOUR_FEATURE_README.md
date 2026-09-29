# Four-feature controlled experiment

Run in Anaconda `lffl`:

```powershell
& C:\Users\46021\.conda\envs\lffl\python.exe src/train_four_feature.py --config src/configs/four_feature.py
```

Or execute `src/train_four_feature.ipynb`. Its default resumes the saved experiment without retraining; set `RESUME_RUN=None` for a new run. Changing configuration or hashed source files requires a new run. The CLI accepts `--resume RUN_DIRECTORY`.

Configuration is a Python dataclass in `four_feature_config.py`; user settings are in `configs/four_feature.py`. The data loader loads the actual previous split manifest. It reads only identities and F1/F2/F3/duration; all four preprocessing statistics are fitted on train and checked against the old corresponding columns. Training workers receive no test data; the GMVAE worker receives no labels, VQ receives only speaker labels, and reference classifiers receive both labels.

VQ uses the same loss/architecture/training protocol as the previous sensitivity run, with four-dimensional input/output. GMVAE is the original `TwoFactorVaDE` with its original statistical objective, optimizer, initialization, pretraining and validation stopping. An equivalent matrix expansion avoids high-dimensional broadcast tensors for its K×K Gaussian components. The tests compare losses and all parameter gradients. GMVAE uses deterministic CUDA for feasibility; set `gmvae_device='cpu'` on systems without CUDA (this creates a distinct recorded execution configuration). No mixed precision is used.

Mandatory matrix: VQ K16/64 × lambda0/1 × five seeds; GMVAE K16/64 × five seeds; five supervised classifiers; deterministic 15-NN once; two raw clusterers × five seeds. Optional extra lambda doses are not part of this run.

`four_feature_report.py RUN_DIRECTORY` generates all paired effects, compact summaries, historical comparisons and plots. Per-seed models, source/config snapshots, predictions, mappings, occupancy, contingencies and feature-wise MSE are retained. Historical VQ accuracy uses the same many-to-one mapping convention. Historical raw clustering includes both frozen Hungarian and many-to-one accuracy. Do not compare 4D and 5D raw MSE directly.

The old manifest preserves split membership, not its old randomized in-memory array order. New conditions share source-row ordering and matched seeds; historical pairs are not identical-initialization interventions. The two unsupervised model families retain their respective training protocols rather than forcing a new common objective or checkpoint rule.
