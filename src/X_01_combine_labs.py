#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

# for configs
import importlib.util
import argparse


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Combine all segmented JVS CSV files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.py"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    LAB_ROOT = Path(config.LAB_ROOT)
    HOME_DIR = Path(config.HOME_DIR)

    HOME_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # Structure:
    # LAB_ROOT / speaker / type / sentence.csv
    csv_files = sorted(LAB_ROOT.glob("*/*/*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    for csv_path in csv_files:

        speaker_id = csv_path.parent.parent.name
        sentence_id = csv_path.stem

        df = pd.read_csv(csv_path)

        df["speaker_id"] = speaker_id
        df["sentence_id"] = sentence_id

        # Example:
        # .../segment/0000.wav -> 0000
        df["segment_id"] = df["path"].apply(
            lambda x: Path(x).stem
        )

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            f"No CSV files found under {LAB_ROOT}"
        )

    combined_df = pd.concat(
        all_rows,
        ignore_index=True
    )

    # ========================================================
    # Save complete metadata
    # ========================================================

    output_path = HOME_DIR / "metadata.csv"

    combined_df.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"Total segments: {len(combined_df)}")
    print(f"Full metadata saved to: {output_path}")

    # ========================================================
    # Save filtered metadata
    # ========================================================

    FILTER_LABELS = getattr(
        config,
        "FILTER_LABELS",
        None
    )

    FILTER_NAME = getattr(
        config,
        "FILTER_NAME",
        None
    )

    if FILTER_LABELS and FILTER_NAME:

        filtered_df = combined_df[
            combined_df["label"].isin(FILTER_LABELS)
        ].copy()

        filtered_output_path = (
            HOME_DIR
            / f"metadata_filtered_{FILTER_NAME}.csv"
        )

        filtered_df.to_csv(
            filtered_output_path,
            index=False
        )

        print(
            f"Filtered labels: "
            f"{sorted(FILTER_LABELS)}"
        )
        print(
            f"Filtered segments: "
            f"{len(filtered_df)}"
        )
        print(
            f"Filtered metadata saved to: "
            f"{filtered_output_path}"
        )

    print()
    print("Finished.")


if __name__ == "__main__":
    main()