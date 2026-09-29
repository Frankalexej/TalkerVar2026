"""Standalone lab baseline, not a neural-model experiment."""
CONFIG = {
    'csv_path': 'metadata_vowels_acoustics.csv',
    'features': ['f0_median_hz','f1_hz','f2_hz','f3_hz','duration_s'],
    'seeds': [42,43,44],
    'fit_sample_seed': 2026,
    'fit_sample_max': 50000,
    'query_large': 1500,
    'query_small': 300,
    'large_group_threshold': 10000,
    'distance_chunk': 32,
    'kmeans_n_init': 10,
    'kmeans_max_iter': 300,
    'kmeans_tol': 1e-4,
    'workers': 2,
    'threads': 1,
    'output_dir': 'lab_projects/vowel_geometry/results',
}
