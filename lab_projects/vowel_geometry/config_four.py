"""Matched all-speaker geometry baseline with F0 completely excluded."""
from pathlib import Path
import runpy

CONFIG = runpy.run_path(str(Path(__file__).with_name('config.py')))['CONFIG'].copy()
CONFIG.update(
    features=['f1_hz', 'f2_hz', 'f3_hz', 'duration_s'],
    output_dir='lab_projects/vowel_geometry/results_four',
)
