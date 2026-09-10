#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import parselmouth

# for configs
import importlib.util
import argparse


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# ============================================================
# Helper: robust formant measurement
# ============================================================

def get_formant_median(
    formant,
    formant_number,
    start_time,
    end_time,
    sample_step,
):
    """
    Sample one formant repeatedly between start_time and end_time
    and return the median of all valid measurements.

    Invalid/undefined Praat estimates are ignored.
    """

    sample_times = np.arange(
        start_time,
        end_time + sample_step / 2,
        sample_step,
    )

    values = []

    for time in sample_times:

        value = formant.get_value_at_time(
            formant_number,
            time,
        )

        if value is not None and np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


# ============================================================
# Extract acoustics from one vowel segment
# ============================================================

def extract_acoustics(
    wav_path,
    pitch_floor,
    pitch_ceiling,
    maximum_formant,
    formant_window_length,
    formant_measure_start,
    formant_measure_end,
    formant_sample_step,
):
    """
    Extract acoustic properties from one segmented vowel WAV.

    F0:
        Median and mean across all voiced pitch frames.

    F1/F2/F3:
        Praat Burg formant analysis.

        Instead of measuring at one exact midpoint, repeatedly
        sample the central portion of the vowel and take the
        median.

        For example, with:
            formant_measure_start = 0.30
            formant_measure_end   = 0.70

        measurements are taken between 30% and 70% of the
        vowel duration.
    """

    sound = parselmouth.Sound(str(wav_path))

    duration = sound.get_total_duration()

    sound_start = sound.get_start_time()
    sound_end = sound.get_end_time()

    # --------------------------------------------------------
    # F0
    # --------------------------------------------------------

    f0_median = np.nan
    f0_mean = np.nan

    try:

        pitch = sound.to_pitch(
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )

        pitch_values = pitch.selected_array[
            "frequency"
        ]

        # Praat represents unvoiced frames as 0 Hz
        voiced_pitch_values = pitch_values[
            pitch_values > 0
        ]

        if len(voiced_pitch_values) > 0:

            f0_median = float(
                np.median(voiced_pitch_values)
            )

            f0_mean = float(
                np.mean(voiced_pitch_values)
            )

    except parselmouth.PraatError:

        # Some vowel segments are too short for reliable pitch
        # analysis at the configured pitch floor.
        #
        # Do NOT reject the vowel: retain NaN for F0 and
        # continue with F1/F2/F3 extraction.
        pass


    # --------------------------------------------------------
    # Formants
    # --------------------------------------------------------

    formant = sound.to_formant_burg(
        maximum_formant=maximum_formant,
        window_length=formant_window_length,
    )

    # Central measurement region
    formant_start_time = (
        sound_start
        + duration * formant_measure_start
    )

    formant_end_time = (
        sound_start
        + duration * formant_measure_end
    )

    # --------------------------------------------------------
    # Median F1/F2/F3 over central region
    # --------------------------------------------------------

    f1 = get_formant_median(
        formant=formant,
        formant_number=1,
        start_time=formant_start_time,
        end_time=formant_end_time,
        sample_step=formant_sample_step,
    )

    f2 = get_formant_median(
        formant=formant,
        formant_number=2,
        start_time=formant_start_time,
        end_time=formant_end_time,
        sample_step=formant_sample_step,
    )

    f3 = get_formant_median(
        formant=formant,
        formant_number=3,
        start_time=formant_start_time,
        end_time=formant_end_time,
        sample_step=formant_sample_step,
    )

    return {
        "duration_s": duration,
        "f0_median_hz": f0_median,
        "f0_mean_hz": f0_mean,
        "f1_hz": f1,
        "f2_hz": f2,
        "f3_hz": f3,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Extract acoustic properties from JVS vowel segments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.py"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    HOME_DIR = Path(config.HOME_DIR)
    REC_ROOT = Path(config.REC_ROOT)

    INPUT_FILENAME = config.INPUT_FILENAME
    OUTPUT_FILENAME = config.OUTPUT_FILENAME

    input_path = (
        HOME_DIR
        / INPUT_FILENAME
    )

    output_path = (
        HOME_DIR
        / OUTPUT_FILENAME
    )


    # --------------------------------------------------------
    # Acoustic-analysis parameters
    # --------------------------------------------------------

    PITCH_FLOOR = getattr(
        config,
        "PITCH_FLOOR",
        60.0,
    )

    PITCH_CEILING = getattr(
        config,
        "PITCH_CEILING",
        500.0,
    )

    MAXIMUM_FORMANT = getattr(
        config,
        "MAXIMUM_FORMANT",
        5500.0,
    )

    FORMANT_WINDOW_LENGTH = getattr(
        config,
        "FORMANT_WINDOW_LENGTH",
        0.025,
    )

    # Portion of vowel over which formants are measured.
    #
    # 0.30 -- 0.70 means the central 40% of the vowel.
    FORMANT_MEASURE_START = getattr(
        config,
        "FORMANT_MEASURE_START",
        0.30,
    )

    FORMANT_MEASURE_END = getattr(
        config,
        "FORMANT_MEASURE_END",
        0.70,
    )

    # Measure formants every 5 ms inside the selected region.
    FORMANT_SAMPLE_STEP = getattr(
        config,
        "FORMANT_SAMPLE_STEP",
        0.005,
    )


    # --------------------------------------------------------
    # Check parameters
    # --------------------------------------------------------

    if not (
        0 <= FORMANT_MEASURE_START
        < FORMANT_MEASURE_END
        <= 1
    ):
        raise ValueError(
            "FORMANT_MEASURE_START and FORMANT_MEASURE_END "
            "must satisfy:\n"
            "0 <= START < END <= 1"
        )


    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{input_path}"
        )

    df = pd.read_csv(
        input_path,
        dtype={
            "speaker_id": str,
            "sentence_id": str,
            "segment_id": str,
        },
    )

    print(f"Metadata: {input_path}")
    print(f"Total vowel segments: {len(df)}")

    print(
        f"Speakers: "
        f"{df['speaker_id'].nunique()}"
    )

    print(
        f"Formant measurement region: "
        f"{FORMANT_MEASURE_START:.0%}"
        f"–"
        f"{FORMANT_MEASURE_END:.0%}"
    )

    print(
        f"Formant sampling interval: "
        f"{FORMANT_SAMPLE_STEP * 1000:.1f} ms"
    )

    print()


    # --------------------------------------------------------
    # New columns
    # --------------------------------------------------------

    acoustic_columns = [
        "duration_s",
        "f0_median_hz",
        "f0_mean_hz",
        "f1_hz",
        "f2_hz",
        "f3_hz",
    ]

    for column in acoustic_columns:
        df[column] = np.nan


    # --------------------------------------------------------
    # Process speaker by speaker
    # --------------------------------------------------------

    speaker_ids = sorted(
        df["speaker_id"].unique()
    )

    INCLUDED_SPEAKERS = getattr(
        config,
        "INCLUDED_SPEAKERS",
        None,
    )

    if INCLUDED_SPEAKERS is not None:

        if isinstance(INCLUDED_SPEAKERS, str):
            INCLUDED_SPEAKERS = [INCLUDED_SPEAKERS]

        speaker_ids = [
            s for s in speaker_ids
            if s in INCLUDED_SPEAKERS
        ]

    total_speakers = len(speaker_ids)

    df = df[
        df["speaker_id"].isin(speaker_ids)
    ].copy()

    print(
        f"Speakers selected ({total_speakers}): "
        f"{speaker_ids}",
        flush=True,
    )

    print()

    for speaker_index, speaker_id in enumerate(
        speaker_ids,
        start=1,
    ):

        speaker_indices = df.index[
            df["speaker_id"] == speaker_id
        ]

        print(
            f"[{speaker_index}/{total_speakers}] "
            f"Processing {speaker_id}: "
            f"{len(speaker_indices)} vowel segments",
            flush=True,
        )

        failed = 0
        f0_missing = 0

        for row_index in speaker_indices:

            relative_path = df.at[
                row_index,
                "path"
            ]

            wav_path = (
                REC_ROOT
                / relative_path
            )

            if not wav_path.exists():

                print(
                    f"  [WARNING] Missing: "
                    f"{wav_path}"
                )

                failed += 1
                continue

            try:

                acoustics = extract_acoustics(
                    wav_path=wav_path,
                    pitch_floor=PITCH_FLOOR,
                    pitch_ceiling=PITCH_CEILING,
                    maximum_formant=MAXIMUM_FORMANT,
                    formant_window_length=FORMANT_WINDOW_LENGTH,
                    formant_measure_start=FORMANT_MEASURE_START,
                    formant_measure_end=FORMANT_MEASURE_END,
                    formant_sample_step=FORMANT_SAMPLE_STEP,
                )

                if np.isnan(
                    acoustics["f0_median_hz"]
                ):
                    f0_missing += 1

                for column, value in acoustics.items():

                    df.at[
                        row_index,
                        column
                    ] = value

            except Exception as e:

                print(
                    f"  [WARNING] Failed: "
                    f"{wav_path}"
                )

                print(
                    f"            {e}"
                )

                failed += 1

        print(
            f"    Finished {speaker_id}: "
            f"{len(speaker_indices) - failed} successfully processed, "
            f"{failed} completely failed, "
            f"{f0_missing} without F0",
            flush=True,
        )

    # --------------------------------------------------------
    # Select output fields
    # --------------------------------------------------------

    INCLUDE_FIELDS = getattr(
        config,
        "INCLUDE_FIELDS",
        None,
    )

    if INCLUDE_FIELDS is not None:

        missing_fields = [
            field for field in INCLUDE_FIELDS
            if field not in df.columns
        ]

        if missing_fields:
            raise ValueError(
                f"These INCLUDE_FIELDS are not present in metadata: "
                f"{missing_fields}"
            )

        df = df[
            INCLUDE_FIELDS
        ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Finished.")

    print(
        f"Saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()