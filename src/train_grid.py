"""python src/train_grid.py --config src/configs/neural_grid.py [--resume RUN_DIR]"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grid_search import run_grid
from src.search_config import load_search_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original VaDE: neural-only grid search")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    run_grid(load_search_config(args.config), resume=args.resume)
