"""Independent within-speaker random splits and an untouched speaker holdout."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import project_path
from src.data import VowelData


def prepare_vq_data(config):
    settings = config.data
    path = project_path(settings.csv_path)
    full = pd.read_csv(path, dtype={'speaker_id': str, 'label': str})
    if full[['speaker_id', 'label']].isna().any().any():
        raise ValueError('Missing speaker/phoneme identity.')
    speakers = sorted(config.training_speakers)
    assert len(set(speakers)) == 10
    available = sorted(full.speaker_id.unique())
    assert set(speakers) <= set(available)
    remaining = sorted(set(available) - set(speakers))
    unseen = sorted(np.random.default_rng(settings.seed).choice(remaining, config.heldout_count, replace=False).tolist())
    frame = full.loc[full.speaker_id.isin(speakers + unseen)].copy()
    frame.insert(0, 'source_row', frame.index)
    frame.reset_index(drop=True, inplace=True)
    parts = {'train': [], 'val': [], 'test_seen': [], 'test_unseen': frame.index[frame.speaker_id.isin(unseen)].tolist()}
    for i, speaker in enumerate(speakers):
        ix = frame.index[frame.speaker_id == speaker].to_numpy()
        # Splitting uses speaker membership and RNG only, never phoneme identity.
        train, holdout = train_test_split(ix, test_size=0.2, random_state=settings.seed + 2*i)
        val, test = train_test_split(holdout, test_size=0.5, random_state=settings.seed + 2*i + 1)
        for name, values in (('train', train), ('val', val), ('test_seen', test)):
            parts[name].extend(values.tolist())
    indices = {p: np.array(ix, dtype=np.int64) for p, ix in parts.items()}
    phonemes = sorted(frame.iloc[indices['train']].label.unique())
    all_phonemes = sorted(frame.label.unique())
    speaker_order = speakers + unseen
    labels = np.column_stack([pd.Categorical(frame.label, categories=phonemes).codes,
                              pd.Categorical(frame.speaker_id, categories=speaker_order).codes]).astype(np.int64)
    raw = frame[list(settings.features)].apply(pd.to_numeric, errors='raise').to_numpy(float)
    raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
    if settings.log_features:
        raw = np.log(raw)
    if np.isnan(raw[indices['train']]).all(0).any():
        raise ValueError('Entire training feature missing.')
    imputer, scaler = SimpleImputer(strategy='median'), StandardScaler()
    scaler.fit(imputer.fit_transform(raw[indices['train']]))
    x = {p: scaler.transform(imputer.transform(raw[ix])).astype('float32') for p, ix in indices.items()}
    metadata = {'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'training_speakers': speakers, 'heldout_speakers': unseen, 'speaker_order': speaker_order,
                'phonemes': phonemes, 'absent_training_phonemes': sorted(set(all_phonemes)-set(phonemes)),
                'features': list(settings.features), 'seed': settings.seed,
                'split_counts': {p: len(ix) for p, ix in indices.items()},
                'phoneme_counts': {p: frame.iloc[ix].label.value_counts().sort_index().to_dict() for p, ix in indices.items()},
                'speaker_counts': {p: frame.iloc[ix].speaker_id.value_counts().sort_index().to_dict() for p, ix in indices.items()},
                'imputation_medians': imputer.statistics_.tolist(), 'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(), 'split_unit': 'vowel token; independently random within each training speaker',
                'missing_or_invalid': {p: dict(zip(settings.features, np.isnan(raw[ix]).sum(0).tolist())) for p, ix in indices.items()}}
    return VowelData(frame, indices, x, {p: labels[ix] for p, ix in indices.items()}, phonemes, speakers, imputer, scaler, metadata)

