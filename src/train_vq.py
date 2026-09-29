"""CLI: python src/train_vq.py --config src/configs/dual_vq.py [--resume DIR]."""
import argparse
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vq_experiment import run_vq_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/configs/dual_vq.py')
    parser.add_argument('--resume')
    args = parser.parse_args()
    result = run_vq_experiment(runpy.run_path(args.config)['CONFIG'], args.resume)
    print(f'COMPLETE: {result}', flush=True)
