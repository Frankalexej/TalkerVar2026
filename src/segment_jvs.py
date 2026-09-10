#!/usr/bin/env python3

from pathlib import Path
import csv
import wave

# for configs
import importlib.util
import argparse


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# ============================================================
# Read monophone alignment
# ============================================================

def read_lab(lab_path):
    """
    Read a JVS monophone .lab file.

    Each line is expected to have:
        start_time end_time label

    Example:
        0.49000 0.55000 m
    """

    segments = []

    with lab_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Unexpected format in {lab_path}, "
                    f"line {line_number}: {line!r}"
                )

            start, end, label = parts

            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "start_str": start,
                    "end_str": end,
                    "label": label,
                }
            )

    return segments


# ============================================================
# Segment WAV
# ============================================================

def segment_wav(wav_path, segments, output_dir, csv_path, relative_base):
    """
    Slice a PCM WAV according to alignment times.

    Uses Python's built-in wave module, so no external
    audio package is required.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    with wave.open(str(wav_path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        compression_type = wav.getcomptype()
        compression_name = wav.getcompname()

        if compression_type != "NONE":
            raise ValueError(
                f"{wav_path} is compressed WAV ({compression_type}); "
                "this script expects ordinary PCM WAV."
            )

        for segment_index, seg in enumerate(segments):

            start_frame = round(seg["start"] * sample_rate)
            end_frame = round(seg["end"] * sample_rate)

            # Prevent very small annotation/rounding differences from
            # exceeding the actual WAV boundaries.
            start_frame = max(0, min(start_frame, n_frames))
            end_frame = max(0, min(end_frame, n_frames))

            if end_frame < start_frame:
                raise ValueError(
                    f"Invalid segment in {wav_path}: "
                    f"{seg['start']} -> {seg['end']}"
                )

            # Give every segment a stable zero-padded index.
            segment_filename = f"{segment_index:04d}.wav"
            segment_path = output_dir / segment_filename

            # Move to the desired position in the source recording.
            wav.setpos(start_frame)

            frames = wav.readframes(end_frame - start_frame)

            # Save segment while preserving the original WAV format.
            with wave.open(str(segment_path), "wb") as out_wav:
                out_wav.setnchannels(n_channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(sample_rate)
                out_wav.setcomptype(
                    compression_type,
                    compression_name,
                )
                out_wav.writeframes(frames)

            relative_path = relative_base / segment_filename

            csv_rows.append(
                {
                    "path": relative_path.as_posix(),
                    "label": seg["label"],
                    "start": seg["start_str"],
                    "end": seg["end_str"],
                }
            )

    # Write one CSV for the entire sentence.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "start", "end"],
        )

        writer.writeheader()
        writer.writerows(csv_rows)


# ============================================================
# Process one speaker / data type
# ============================================================

def process_data_type(
    speaker_dir,
    data_type,
    JVS_ROOT,
    REC_ROOT,
    LAB_ROOT,
):
    speaker = speaker_dir.name

    type_dir = speaker_dir / data_type
    mon_dir = type_dir / "lab" / "mon"
    wav_dir = type_dir / "wav24kHz16bit"

    if not mon_dir.is_dir():
        print(f"[SKIP] Missing mon directory: {mon_dir}")
        return 0

    if not wav_dir.is_dir():
        print(f"[SKIP] Missing WAV directory: {wav_dir}")
        return 0

    lab_files = sorted(mon_dir.glob("*.lab"))

    processed = 0

    for lab_path in lab_files:

        # Sentence ID is the shared basename of xxx.lab and xxx.wav.
        sentence = lab_path.stem

        wav_path = wav_dir / f"{sentence}.wav"

        if not wav_path.exists():
            print(
                f"[WARNING] No matching WAV for "
                f"{lab_path.relative_to(JVS_ROOT)}"
            )
            continue

        segments = read_lab(lab_path)

        # Example:
        # data/rec/jvs001/parallel100/BASIC5000_0001/segment/
        output_dir = (
            REC_ROOT
            / speaker
            / data_type
            / sentence
            / "segment"
        )

        # Example:
        # data/lab/jvs001/parallel100/BASIC5000_0001.csv
        csv_path = (
            LAB_ROOT
            / speaker
            / data_type
            / f"{sentence}.csv"
        )

        # Stored in CSV relative to data/rec/
        relative_base = (
            Path(speaker)
            / data_type
            / sentence
            / "segment"
        )

        segment_wav(
            wav_path=wav_path,
            segments=segments,
            output_dir=output_dir,
            csv_path=csv_path,
            relative_base=relative_base,
        )

        processed += 1

    return processed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Training script with config path'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.py'
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Read paths from config
    JVS_ROOT = Path(config.JVS_ROOT)
    REC_ROOT = Path(config.REC_ROOT)
    LAB_ROOT = Path(config.LAB_ROOT)

    DATA_TYPES = set(config.DATA_TYPES)
    SPEAKER_DIRS = config.SPEAKER_DIRS

    if not JVS_ROOT.is_dir():
        raise FileNotFoundError(
            f"JVS dataset not found at:\n{JVS_ROOT}"
        )

    REC_ROOT.mkdir(parents=True, exist_ok=True)
    LAB_ROOT.mkdir(parents=True, exist_ok=True)

    if not SPEAKER_DIRS:
        speaker_dirs = sorted(
            p for p in JVS_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("jvs")
        )
    else: 
        speaker_dirs = sorted(
            JVS_ROOT / speaker
            for speaker in SPEAKER_DIRS
            if (JVS_ROOT / speaker).is_dir()
        )

    print(f"JVS root: {JVS_ROOT}")
    print(f"Speakers found: {len(speaker_dirs)}")
    print(f"Data types: {sorted(DATA_TYPES)}")
    print()

    total_sentences = 0

    for speaker_dir in speaker_dirs:

        speaker = speaker_dir.name
        speaker_total = 0

        for data_type in sorted(DATA_TYPES):

            # Not every speaker/data type necessarily needs to exist.
            if not (speaker_dir / data_type).is_dir():
                continue

            count = process_data_type(
                speaker_dir,
                data_type,
                JVS_ROOT,
                REC_ROOT,
                LAB_ROOT,
            )

            speaker_total += count
            total_sentences += count

        print(
            f"[{speaker}] processed "
            f"{speaker_total} sentences"
        )

    print()
    print("Finished.")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Segmented WAV root: {REC_ROOT}")
    print(f"CSV label root:     {LAB_ROOT}")


if __name__ == "__main__":
    main()