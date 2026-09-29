"""Anaconda/Slurm entry point for the four-feature experiment."""
import argparse
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.four_feature_experiment import run_four

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='src/configs/four_feature.py')
    p.add_argument('--resume')
    args = p.parse_args()
    print('COMPLETE:',run_four(runpy.run_path(args.config)['CONFIG'],args.resume),flush=True)
