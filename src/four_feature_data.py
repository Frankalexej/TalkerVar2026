"""Load the actual saved token manifest, never regenerate a random split."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import project_path
from src.data import VowelData


def prepare_four_feature_data(config):
    ref = project_path(config.reference_run)
    old = json.loads((ref/'data.json').read_text())
    path = project_path(config.data.csv_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == old['csv_sha256'], 'Source dataset changed'
    features = list(config.data.features)
    assert features == ['f1_hz','f2_hz','f3_hz','duration_s']
    manifest = pd.read_csv(ref/'split_manifest.csv',dtype={'speaker_id':str,'label':str})
    assert manifest.source_row.is_unique
    # F0 is never loaded as a column, used in cleaning, or exposed to models.
    raw = pd.read_csv(path,usecols=['speaker_id','label']+features,dtype={'speaker_id':str,'label':str})
    frame = raw.iloc[manifest.source_row.to_numpy()].copy().reset_index(drop=True)
    frame.insert(0,'source_row',manifest.source_row.to_numpy())
    pd.testing.assert_frame_equal(frame[['source_row','speaker_id','label']],manifest[['source_row','speaker_id','label']])
    parts = ('train','val','test_seen','test_unseen')
    indices = {p:np.flatnonzero(manifest.split.eq(p)) for p in parts}
    assert sum(map(len,indices.values())) == len(frame)
    speakers, unseen = old['training_speakers'],old['heldout_speakers']
    for p in parts:
        assert set(frame.iloc[indices[p]].speaker_id) == set(unseen if p=='test_unseen' else speakers)
    phonemes = sorted(frame.iloc[indices['train']].label.unique())
    y = np.column_stack([pd.Categorical(frame.label,categories=phonemes).codes,
                         pd.Categorical(frame.speaker_id,categories=speakers+unseen).codes]).astype('int64')
    values = frame[features].to_numpy(float)
    values[~np.isfinite(values)|(values<=0)] = np.nan
    assert not config.data.log_features
    imputer,scaler = SimpleImputer(strategy='median'),StandardScaler()
    scaler.fit(imputer.fit_transform(values[indices['train']]))
    # Equality to the corresponding old statistics guards accidental leakage.
    for new,field in ((imputer.statistics_,'imputation_medians'),(scaler.mean_,'scaler_mean'),(scaler.scale_,'scaler_scale')):
        np.testing.assert_allclose(new,np.asarray(old[field])[1:],rtol=1e-12,atol=1e-12)
    x = {p:scaler.transform(imputer.transform(values[ix])).astype('float32') for p,ix in indices.items()}
    metadata = {**old,'features':features,'csv_sha256':digest,
                'manifest_sha256':hashlib.sha256((ref/'split_manifest.csv').read_bytes()).hexdigest(),
                'split_source':str(ref/'split_manifest.csv'),
                'imputation_medians':imputer.statistics_.tolist(),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
                'missing_or_invalid':{p:dict(zip(features,np.isnan(values[ix]).sum(0).tolist())) for p,ix in indices.items()}}
    return VowelData(frame,indices,x,{p:y[ix] for p,ix in indices.items()},phonemes,speakers,imputer,scaler,metadata)

