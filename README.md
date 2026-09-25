# TalkerVar2026
Talker variation and acquisition trajectory

## Vowel model experiments

See [the training notebook](src/train_vowels.ipynb) and
[the project guide](src/README.md). Edit `src/configs/vowels.py`, then run:

```bash
python src/train.py --config src/configs/vowels.py
```

Use an Anaconda environment with the dependencies in `src/environment.yml`.
The tested local environment is `lffl`. The CSV stays in this repository's root.

The paper-inspired extension with one shared continuous auxiliary variable is in
[a separate notebook](src/train_shared_w.ipynb), configured by
`src/configs/shared_w.py`. It adds `SharedWGMVAE` alongside the original models.

For a neural-only grid search of the original VaDE, use
[the grid-search notebook](src/search_neural_grid.ipynb) or run
`python src/train_grid.py --config src/configs/neural_grid.py`.
