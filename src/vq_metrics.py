"""Test-only descriptive metrics; Hungarian mappings are fitted on train only."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.evaluation import fit_alignment
from src.experiment import write_json
from src.vq_training import predict_vq, load_vq, perplexity


def contingency(codes, labels, size, classes):
    result = np.zeros((size, len(classes)), dtype=int)
    for col, label in enumerate(classes):
        result[:, col] = np.bincount(codes[labels == label], minlength=size)
    return result


def code_metrics(codes, labels, size):
    matrix = contingency(codes, labels, size, np.unique(labels))
    return {'nmi': float(normalized_mutual_info_score(labels, codes)),
            'ari': float(adjusted_rand_score(labels, codes)),
            'purity': float(matrix.max(1).sum()/len(labels))}


def evaluate_codes(kp, ks, yp, ys, nphonemes, pmapping, smapping, unseen=False):
    result = {'phoneme_accuracy': float(np.mean(pmapping[kp] == yp))}
    for branch, codes, size in (('phi', kp, nphonemes), ('s', ks, 10)):
        for factor, labels in (('phi', yp), ('s', ys)):
            result.update({f'{branch}_{factor}_{k}': v for k, v in code_metrics(codes, labels, size).items()})
        counts = np.bincount(codes, minlength=size)
        result[branch+'_active_codes'] = int((counts > 0).sum())
        result[branch+'_perplexity'] = perplexity(codes, size)
        result[branch+'_dominant_fraction'] = float(counts.max()/counts.sum())
    if not unseen:
        result['speaker_code_accuracy'] = float(np.mean(smapping[ks] == ys))
    return result


def evaluate_vq(directory, config, data, condition, seed):
    directory = Path(directory)
    model, checkpoint = load_vq(directory, config)
    predictions = {p: predict_vq(model, x) for p, x in data.x.items()}
    yp, ys = data.y['train'].T
    pmapping = fit_alignment(yp, predictions['train']['k_phi'], len(data.phonemes))
    smapping = fit_alignment(ys, predictions['train']['k_s'], 10)
    write_json(directory/'mappings.json', {'fit_split': 'train', 'phoneme': pmapping.tolist(), 'speaker': smapping.tolist()})
    train_z = {branch: predictions['train']['z_'+branch].mean(0) for branch in ('phi', 's')}
    rows = []
    for part in ('test_seen', 'test_unseen'):
        out, x = predictions[part], data.x[part]
        yp, ys = data.y[part].T
        metrics = evaluate_codes(out['k_phi'], out['k_s'], yp, ys, len(data.phonemes), pmapping, smapping, part == 'test_unseen')
        if part == 'test_seen' and condition == 'A':
            metrics['speaker_classifier_accuracy'] = float(np.mean(out['speaker_logits'].argmax(1) == ys))
        error = np.square(out['reconstruction']-x)
        metrics['mse'] = float(error.mean())
        for feature, value in zip(config.data.features, error.mean(0)):
            metrics['mse_'+feature] = float(value)
        for branch in ('phi', 's'):
            z = {b: torch.from_numpy(out['z_'+b]) for b in ('phi', 's')}
            z[branch] = torch.from_numpy(np.broadcast_to(train_z[branch], z[branch].shape).copy())
            with torch.no_grad():
                reconstructed = torch.cat([model.decoder(torch.cat([p, s], 1))
                                           for p, s in zip(z['phi'].split(2048), z['s'].split(2048))]).numpy()
            metrics['ablate_'+branch+'_mse'] = float(np.square(reconstructed-x).mean())
            metrics['ablate_'+branch+'_delta'] = metrics['ablate_'+branch+'_mse'] - metrics['mse']
        for branch, codes, size in (('phi', out['k_phi'], len(data.phonemes)), ('s', out['k_s'], 10)):
            counts = np.bincount(codes, minlength=size)
            write_json(directory/f'{part}_{branch}_usage.json', {'counts': counts.tolist(), 'unused_codes': np.flatnonzero(counts == 0).tolist(),
                         'rare_codes_below_1pct': np.flatnonzero(counts/len(codes) < 0.01).tolist()})
            for factor, labels in (('phi', yp), ('s', ys)):
                classes = np.unique(labels)
                names = [data.phonemes[i] if factor == 'phi' and i >= 0 else
                         ('unknown_phoneme' if factor == 'phi' else data.metadata['speaker_order'][i]) for i in classes]
                counts = contingency(codes, labels, size, classes)
                pd.DataFrame(counts, columns=names).to_csv(directory/f'{part}_{branch}_by_{factor}.csv', index_label='code')
                if factor == 's':
                    pd.DataFrame(counts/np.maximum(counts.sum(0), 1), columns=names).to_csv(
                        directory/f'{part}_{branch}_distribution_by_speaker.csv', index_label='code')
        np.savez_compressed(directory/f'{part}_representations.npz', **out, true_phoneme=yp, true_speaker=ys,
                            source_row=data.frame.iloc[data.indices[part]].source_row.to_numpy())
        rows.append({'method': condition, 'seed': seed, 'split': part, **metrics})
    # PCA is fit on train representations only; plotting uses this saved cache.
    np.savez_compressed(directory/'train_representations.npz', h_phi=predictions['train']['h_phi'], h_s=predictions['train']['h_s'])
    write_json(directory/'test_metrics.json', rows)
    return rows
