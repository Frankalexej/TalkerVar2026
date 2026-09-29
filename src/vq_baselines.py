"""Matched-split reruns of existing baseline implementations."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from src.config import ExperimentConfig, ModelConfig, SharedWModelConfig, TrainConfig
from src.models import TwoFactorVaDE, SharedWGMVAE, VowelClassifier
from src.training import fit_neural, seed_everything
from src.evaluation import predict_neural, fit_alignment
from src.experiment import write_json
from src.vq_metrics import evaluate_codes


def train_baseline(config, cache_path, directory, seed, method):
    directory = Path(directory)
    if (directory/'trained.json').exists():
        return json.loads((directory/'trained.json').read_text())
    directory.mkdir(parents=True, exist_ok=True)
    data = SimpleNamespace(**joblib.load(cache_path))
    assert set(data.x) == set(data.y) == {'train', 'val'}
    with (directory/'training.log').open('w') as stream, redirect_stdout(stream), redirect_stderr(stream), threadpool_limits(config.threads):
        seed_everything(seed, config.threads)
        settings = SharedWModelConfig() if method == 'gmvae_shared_w' else ModelConfig()
        cfg = ExperimentConfig(model=settings, training=TrainConfig(seed=seed, num_threads=config.threads))
        write_json(directory/'config.json', asdict(cfg))
        if method == 'knn':
            model = KNeighborsClassifier(n_neighbors=15, n_jobs=config.threads).fit(data.x['train'], data.y['train'])
            joblib.dump(model, directory/'best.joblib')
            details = {'seed': seed, 'method': method}
        else:
            cls = {'gmvae': TwoFactorVaDE, 'gmvae_shared_w': SharedWGMVAE, 'classifier': VowelClassifier}[method]
            model = cls(5, len(data.phonemes), 10, settings)
            model, details = fit_neural(model, method, data, cfg, directory)
            torch.save({'state_dict': model.state_dict(), 'method': method}, directory/'best.pt')
        write_json(directory/'trained.json', details)
    return details


def evaluate_baseline(config, data, directory, method, seed):
    directory = Path(directory)
    if method == 'knn':
        model = joblib.load(directory/'best.joblib')
        raw = {p: model.predict(data.x[p]) for p in ('train', 'test_seen', 'test_unseen')}
    else:
        settings = SharedWModelConfig() if method == 'gmvae_shared_w' else ModelConfig()
        cls = {'gmvae': TwoFactorVaDE, 'gmvae_shared_w': SharedWGMVAE, 'classifier': VowelClassifier}[method]
        model = cls(5, len(data.phonemes), 10, settings)
        model.load_state_dict(torch.load(directory/'best.pt', weights_only=True)['state_dict'])
        model.eval()
        raw = {p: predict_neural(model, method, data.x[p], 512, seed+200)[0] for p in ('train', 'test_seen', 'test_unseen')}
    mixture = method.startswith('gmvae')
    maps = [fit_alignment(data.y['train'][:,i], raw['train'][:,i], k) if mixture else np.arange(k)
            for i, k in enumerate((len(data.phonemes), 10))]
    write_json(directory/'mappings.json', {'fit_split': 'train', 'phoneme': maps[0].tolist(), 'speaker': maps[1].tolist()})
    rows = []
    for part in ('test_seen', 'test_unseen'):
        p, s = raw[part].T
        yp, ys = data.y[part].T
        metrics = evaluate_codes(p, s, yp, ys, len(data.phonemes), *maps, unseen=part == 'test_unseen')
        if not mixture:
            if part == 'test_seen':
                metrics['speaker_classifier_accuracy'] = metrics.pop('speaker_code_accuracy')
            # These are classifier outputs, not codebooks; retain only accuracies.
            metrics = {k:v for k,v in metrics.items() if k in ('phoneme_accuracy', 'speaker_classifier_accuracy')}
        else:
            with torch.no_grad():
                reconstructed = torch.cat([model.decoder(model.encode(torch.from_numpy(b))[0])
                                            for b in np.array_split(data.x[part], max(1, int(np.ceil(len(p)/2048))))]).numpy()
            error = np.square(reconstructed-data.x[part])
            metrics['mse'] = float(error.mean())
            metrics.update({'mse_'+f:float(v) for f,v in zip(config.data.features, error.mean(0))})
        pd.DataFrame({'source_row': data.frame.iloc[data.indices[part]].source_row.to_numpy(),
                      'true_phoneme': yp, 'true_speaker': ys, 'raw_phi': p, 'raw_s': s,
                      'mapped_phoneme': maps[0][p]}).to_csv(directory/f'{part}_predictions.csv', index=False)
        rows.append({'method': method, 'seed': seed, 'split': part, **metrics})
    write_json(directory/'test_metrics.json', rows)
    return rows
