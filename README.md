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
