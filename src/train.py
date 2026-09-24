"""CLI: python src/train.py --config src/configs/vowels.py"""
import argparse
from pathlib import Path
import sys

# Support both `python src/train.py` and `python -m src.train`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train vowel GMVAE and supervised baselines.")
    parser.add_argument("--config", required=True, help="Trusted Python file defining CONFIG")
    args = parser.parse_args()
    run_experiment(load_config(args.config))


if __name__ == "__main__":
    main()
