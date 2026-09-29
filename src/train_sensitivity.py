"""Run the two follow-up experiments from a Python configuration."""
import argparse
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.sensitivity_experiment import run_sensitivity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='src/configs/vq_sensitivity.py')
    parser.add_argument('--resume')
    args = parser.parse_args()
    print('COMPLETE:',run_sensitivity(runpy.run_path(args.config)['CONFIG'],args.resume),flush=True)
