#!/usr/bin/env python3
"""Extract vowel acoustics from original JVS utterances, using existing metadata.

Run: python X_02_extract_vowel_acoustics_sentence.py --config config.py
The input path column remains a path relative to REC_ROOT, as produced by
segment_jvs.py; no segmented WAV files are opened by this script.
"""

import argparse
import importlib.util
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import parselmouth


ACOUSTIC_COLUMNS = [
    "duration_s", "f0_median_hz", "f0_mean_hz", "f1_hz", "f2_hz", "f3_hz",
]
NUMERIC_QC_COLUMNS = [
    "f0_n_frames", "f0_n_voiced_frames", "f0_voiced_ratio",
    "formant_n_frames", "f1_n_valid_frames", "f2_n_valid_frames",
    "f3_n_valid_frames",
]
TEXT_QC_COLUMNS = [
    "data_type", "sentence_wav_path", "f0_status", "formant_status",
    "extraction_error",
]


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def sentence_key(row):
    """Decode the exact relative path format emitted by segment_jvs.py."""
    if pd.isna(row["path"]):
        raise ValueError("Missing segment path")
    p = PurePosixPath(str(row["path"]).replace("\\", "/"))
    parts = p.parts
    if p.is_absolute() or len(parts) != 5 or parts[3] != "segment":
        raise ValueError(f"Unexpected segment path: {p}")
    if any(part in {"..", "."} for part in parts) or p.suffix.lower() != ".wav":
        raise ValueError(f"Invalid segment path: {p}")
    speaker, data_type, sentence, _, _ = parts
    for column, expected in (("speaker_id", speaker), ("sentence_id", sentence),
                             ("segment_id", p.stem)):
        if pd.isna(row[column]) or str(row[column]) != expected:
            raise ValueError(f"{column} disagrees with segment path: {p}")
    return speaker, data_type, sentence


def sentence_path(key):
    speaker, data_type, sentence = key
    return Path(speaker) / data_type / "wav24kHz16bit" / f"{sentence}.wav"


def analyse_sentence(sound, settings):
    """Run each Praat analysis once, preserving independent failures."""
    pitch = formant = None
    pitch_error = formant_error = ""
    try:
        pitch = sound.to_pitch_ac(
            time_step=settings["pitch_step"],
            pitch_floor=settings["pitch_floor"],
            pitch_ceiling=settings["pitch_ceiling"],
        )
    except parselmouth.PraatError as exc:
        pitch_error = f"Pitch: {exc}"
    try:
        formant = sound.to_formant_burg(
            time_step=settings["formant_step"],
            maximum_formant=settings["maximum_formant"],
            window_length=settings["formant_window"],
        )
    except parselmouth.PraatError as exc:
        formant_error = f"Formants: {exc}"
    return pitch, formant, pitch_error, formant_error


def valid_interval(start, end, sound, tolerance):
    """Reject invalid annotations; clip only tiny endpoint discrepancies."""
    start, end = float(start), float(end)
    if not (np.isfinite(start) and np.isfinite(end) and start < end):
        raise ValueError(f"Invalid annotation interval: {start}, {end}")
    lo, hi = sound.get_start_time(), sound.get_end_time()
    if start < lo - tolerance or end > hi + tolerance:
        raise ValueError(f"Interval {start:.6f}–{end:.6f} outside WAV {lo:.6f}–{hi:.6f}")
    start, end = max(lo, start), min(hi, end)
    if start >= end:
        raise ValueError("Empty interval after clipping to WAV boundaries")
    return start, end


def extract_acoustics(pitch, formant, start, end, settings):
    """Select native Praat frames by their center times, not by window edges."""
    duration = end - start
    result = {name: np.nan for name in ACOUSTIC_COLUMNS + NUMERIC_QC_COLUMNS}
    result["duration_s"] = duration
    result["f0_status"] = "analysis_failed"
    result["formant_status"] = "analysis_failed"

    if pitch is not None:
        times = pitch.xs()
        values = pitch.selected_array["frequency"]
        selected = values[(times >= start) & (times < end)]
        voiced = selected[np.isfinite(selected) & (selected > 0)]
        n_frames, n_voiced = len(selected), len(voiced)
        result.update(f0_n_frames=n_frames, f0_n_voiced_frames=n_voiced,
                      f0_voiced_ratio=n_voiced / n_frames if n_frames else np.nan)
        if n_voiced:
            result.update(f0_median_hz=float(np.median(voiced)),
                          f0_mean_hz=float(np.mean(voiced)), f0_status="success")
        else:
            result["f0_status"] = "no_voiced_frames" if n_frames else "no_frames"

    if formant is not None:
        lo = start + duration * settings["measure_start"]
        hi = start + duration * settings["measure_end"]
        times = formant.xs()
        times = times[(times >= lo) & (times <= hi) & (times < end)]
        result["formant_n_frames"] = len(times)
        counts = []
        for number in (1, 2, 3):
            values = np.asarray([
                formant.get_value_at_time(number, float(t)) for t in times
            ], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            counts.append(len(values))
            result[f"f{number}_n_valid_frames"] = len(values)
            if len(values):
                result[f"f{number}_hz"] = float(np.median(values))
        if not len(times):
            result["formant_status"] = "no_frames"
        elif all(counts):
            result["formant_status"] = "success"
        elif any(counts):
            result["formant_status"] = "partial"
        else:
            result["formant_status"] = "no_valid_formants"
    return result


def get_settings(config):
    settings = {
        "pitch_floor": float(getattr(config, "PITCH_FLOOR", 60.0)),
        "pitch_ceiling": float(getattr(config, "PITCH_CEILING", 500.0)),
        "pitch_step": float(getattr(config, "PITCH_TIME_STEP", 0.005)),
        "maximum_formant": float(getattr(config, "MAXIMUM_FORMANT", 5500.0)),
        "formant_window": float(getattr(config, "FORMANT_WINDOW_LENGTH", 0.025)),
        # Old FORMANT_SAMPLE_STEP now specifies spacing of actual Burg frames
        # unless the more explicit new setting is supplied.
        "formant_step": float(getattr(config, "FORMANT_TIME_STEP",
                                     getattr(config, "FORMANT_SAMPLE_STEP", 0.005))),
        "measure_start": float(getattr(config, "FORMANT_MEASURE_START", 0.30)),
        "measure_end": float(getattr(config, "FORMANT_MEASURE_END", 0.70)),
        "boundary_tolerance": float(getattr(config, "BOUNDARY_TOLERANCE_S", 0.001)),
    }
    if not all(np.isfinite(v) for v in settings.values()):
        raise ValueError("Analysis settings must be finite")
    if not 0 < settings["pitch_floor"] < settings["pitch_ceiling"]:
        raise ValueError("Require 0 < PITCH_FLOOR < PITCH_CEILING")
    if not 0 <= settings["measure_start"] < settings["measure_end"] <= 1:
        raise ValueError("Require 0 <= FORMANT_MEASURE_START < FORMANT_MEASURE_END <= 1")
    for key in ("pitch_step", "maximum_formant", "formant_window", "formant_step"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if settings["boundary_tolerance"] < 0:
        raise ValueError("BOUNDARY_TOLERANCE_S must be nonnegative")
    return settings


def mark_failure(df, indices, status, message):
    df.loc[indices, "f0_status"] = status
    df.loc[indices, "formant_status"] = status
    df.loc[indices, "extraction_error"] = str(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.py")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = get_settings(config)
    home = Path(config.HOME_DIR)
    root = Path(config.JVS_ROOT)
    input_path = home / config.INPUT_FILENAME
    output_path = home / config.OUTPUT_FILENAME
    if not root.is_dir():
        raise FileNotFoundError(f"JVS_ROOT not found: {root}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output filenames must be different")

    df = pd.read_csv(input_path, dtype={
        "speaker_id": str, "sentence_id": str, "segment_id": str, "path": str,
    })
    required = {"path", "start", "end", "speaker_id", "sentence_id", "segment_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    included = getattr(config, "INCLUDED_SPEAKERS", None)
    if included is not None:
        if isinstance(included, str):
            included = [included]
        df = df[df["speaker_id"].isin(included)].copy()
    df = df.reset_index(drop=True)

    for column in ACOUSTIC_COLUMNS + NUMERIC_QC_COLUMNS:
        df[column] = np.nan
    for column in TEXT_QC_COLUMNS:
        df[column] = ""
    df["f0_status"] = "not_processed"
    df["formant_status"] = "not_processed"

    # Group by original WAV, not by sentence ID alone; keep input row order.
    groups = {}
    for i, row in df.iterrows():
        try:
            key = sentence_key(row)
        except ValueError as exc:
            mark_failure(df, [i], "invalid_metadata", exc)
            continue
        df.at[i, "data_type"] = key[1]
        df.at[i, "sentence_wav_path"] = sentence_path(key).as_posix()
        groups.setdefault(key, []).append(i)

    print(f"Metadata: {input_path}\nVowel tokens: {len(df)}\nSentences: {len(groups)}")
    print(f"Analysis settings: {settings}", flush=True)
    by_speaker = {}
    for key, indices in groups.items():
        by_speaker.setdefault(key[0], []).append((key, indices))
    for speaker_number, (speaker, utterances) in enumerate(sorted(by_speaker.items()), 1):
        speaker_indices = []
        print(f"[{speaker_number}/{len(by_speaker)}] {speaker}: {len(utterances)} sentences", flush=True)
        for key, indices in utterances:
            speaker_indices.extend(indices)
            wav_path = root / sentence_path(key)
            if not wav_path.is_file():
                mark_failure(df, indices, "missing_wav", f"Missing original WAV: {wav_path}")
                print(f"  [WARNING] Missing original WAV: {wav_path}", flush=True)
                continue
            try:
                sound = parselmouth.Sound(str(wav_path))
            except (parselmouth.PraatError, OSError) as exc:
                mark_failure(df, indices, "wav_read_failed", exc)
                print(f"  [WARNING] Cannot read {wav_path}: {exc}", flush=True)
                continue
            pitch, formant, pitch_error, formant_error = analyse_sentence(sound, settings)
            error = " | ".join(e for e in (pitch_error, formant_error) if e)
            if error:
                print(f"  [WARNING] {wav_path}: {error}", flush=True)
            for i in indices:
                try:
                    start, end = valid_interval(df.at[i, "start"], df.at[i, "end"],
                                                sound, settings["boundary_tolerance"])
                except (ValueError, TypeError) as exc:
                    mark_failure(df, [i], "invalid_interval", exc)
                    continue
                result = extract_acoustics(pitch, formant, start, end, settings)
                for column, value in result.items():
                    df.at[i, column] = value
                df.at[i, "extraction_error"] = error
        statuses = df.loc[speaker_indices, "f0_status"].value_counts().to_dict()
        print(f"  F0: {statuses}", flush=True)

    print("\nF0 status counts:\n" + df["f0_status"].value_counts().to_string())
    print("\nFormant status counts:\n" + df["formant_status"].value_counts().to_string())

    # Respect the original output-column selection; keep a separate full QC CSV
    # when requested so missing measurements can always be investigated.
    fields = getattr(config, "INCLUDE_FIELDS", None)
    if fields is not None:
        missing = set(fields).difference(df.columns)
        if missing:
            raise ValueError(f"Unknown INCLUDE_FIELDS: {sorted(missing)}")
        output_df = df[list(fields)]
    else:
        output_df = df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(config, "SAVE_ACOUSTIC_QC", True):
        qc_path = output_path.with_name(output_path.stem + "_qc.csv")
        if qc_path.resolve() == input_path.resolve():
            raise ValueError("QC output would overwrite input metadata")
        df.to_csv(qc_path, index=False)
        print(f"Full diagnostic CSV: {qc_path}")
    output_df.to_csv(output_path, index=False)
    print(f"Saved acoustic metadata: {output_path}")


if __name__ == "__main__":
    main()
