"""Train from an acoustics/speaker-only cache, with common A/B initialization."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from src.experiment import write_json
from src.training import seed_everything
from src.vq_models import DualVQVAE


def perplexity(indices, size):
    counts = np.bincount(indices, minlength=size)
    p = counts/counts.sum()
    return float(np.exp(-(p[p > 0]*np.log(p[p > 0])).sum()))


@torch.no_grad()
def predict_vq(model, x, batch_size=2048):
    model.eval()
    saved = {}
    for start in range(0, len(x), batch_size):
        out = model(torch.from_numpy(x[start:start+batch_size]))
        for key, value in out.items():
            if value.ndim:
                saved.setdefault(key, []).append(value.numpy())
    return {k: np.concatenate(v) for k, v in saved.items()}


def state_digest(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def initialize(model, x, config, seed):
    # Common continuous autoencoder initialization uses only training acoustics.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1000)
    x = torch.from_numpy(x)
    rows = []
    for epoch in range(config.pretrain_epochs):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for ix in order.split(config.batch_size):
            batch = x[ix]
            reconstructed = model.decoder(torch.cat([model.encoder_phi(batch), model.encoder_s(batch)], 1))
            loss = (reconstructed-batch).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total += loss.item()*len(ix)
        rows.append({'epoch': epoch+1, 'reconstruction': total/len(x)})
    with torch.no_grad():
        for i, (encoder, vq) in enumerate(((model.encoder_phi, model.vq_phi), (model.encoder_s, model.vq_s))):
            h = torch.cat([encoder(b) for b in x.split(2048)]).numpy()
            centers = KMeans(n_clusters=vq.embedding.num_embeddings, n_init=10, random_state=seed+i).fit(h).cluster_centers_
            vq.embedding.weight.copy_(torch.from_numpy(centers))
    return rows


def train_vq(config, cache_path, directory, seed, speaker_weight, epochs):
    directory = Path(directory)
    if (directory/'validation.json').exists():
        return json.loads((directory/'validation.json').read_text())
    directory.mkdir(parents=True, exist_ok=True)
    data = joblib.load(cache_path)
    assert set(data) == {'x', 'speaker', 'n_phonemes'}
    assert set(data['x']) == set(data['speaker']) == {'train', 'val'}
    started = time.perf_counter()
    with (directory/'training.log').open('w') as stream, redirect_stdout(stream), redirect_stderr(stream), threadpool_limits(config.threads):
        seed_everything(seed, config.threads)
        model = DualVQVAE(data['n_phonemes'], config)
        init_rows = initialize(model, data['x']['train'], config, seed)
        pd.DataFrame(init_rows).to_csv(directory/'pretraining.csv', index=False)
        init_hash = state_digest(model)
        torch.save(model.state_dict(), directory/'initial.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        xs = {p: torch.from_numpy(a) for p, a in data['x'].items()}
        ys = {p: torch.from_numpy(a) for p, a in data['speaker'].items()}
        generator = torch.Generator().manual_seed(seed+2000)
        best, best_state, records, best_epoch = float('inf'), None, [], 0
        order_hash = hashlib.sha256()
        for epoch in range(1, epochs+1):
            record = {'epoch': epoch}
            for part in ('train', 'val'):
                training = part == 'train'
                model.train(training)
                order = torch.randperm(len(xs[part]), generator=generator) if training else torch.arange(len(xs[part]))
                if training:
                    order_hash.update(order.numpy().tobytes())
                totals, codes = {}, {'phi': [], 's': []}
                with torch.set_grad_enabled(training):
                    for ix in order.split(config.batch_size):
                        loss, components, out = model.loss(xs[part][ix], ys[part][ix], speaker_weight)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite VQ loss')
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                            optimizer.step()
                        for k, v in components.items():
                            totals[k] = totals.get(k, 0.0) + v.item()*len(ix)
                        for branch in codes:
                            codes[branch].extend(out['k_'+branch].detach().numpy().tolist())
                record.update({part+'_'+k: v/len(order) for k, v in totals.items()})
                for branch, size in (('phi', data['n_phonemes']), ('s', 10)):
                    record[f'{part}_perplexity_{branch}'] = perplexity(codes[branch], size)
            records.append(record)
            # Same selection criterion and fixed training budget for A and B.
            if record['val_reconstruction'] < best:
                best, best_epoch = record['val_reconstruction'], epoch
                best_state = deepcopy(model.state_dict())
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch {epoch}: val MSE={record['val_reconstruction']:.5f}", flush=True)
            pd.DataFrame(records).to_csv(directory/'history.csv', index=False)
        model.load_state_dict(best_state)
        out = predict_vq(model, data['x']['val'])
        unused = 1-len(np.unique(out['k_phi']))/data['n_phonemes'] + 1-len(np.unique(out['k_s']))/10
        nmi = float(normalized_mutual_info_score(data['speaker']['val'], out['k_s']))
        mse = float(np.square(out['reconstruction']-data['x']['val']).mean())
        result = {'seed': seed, 'lambda_s': speaker_weight, 'epochs': epochs, 'best_epoch': best_epoch,
                  'validation_mse': mse, 'validation_speaker_nmi': float(nmi), 'unused_fraction_sum': unused,
                  'selection_score': mse + 0.1*(1-nmi) + 0.05*unused,
                  'initial_state_sha256': init_hash, 'minibatch_order_sha256': order_hash.hexdigest(),
                  'parameter_count': sum(p.numel() for p in model.parameters()),
                  'elapsed_seconds': time.perf_counter()-started}
        torch.save({'state_dict': model.state_dict(), 'n_phonemes': data['n_phonemes'],
                    'config': asdict(config), 'training': result}, directory/'best.pt')
        write_json(directory/'validation.json', result)
    return result


def load_vq(directory, config):
    checkpoint = torch.load(Path(directory)/'best.pt', map_location='cpu', weights_only=True)
    model = DualVQVAE(checkpoint['n_phonemes'], config)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint
