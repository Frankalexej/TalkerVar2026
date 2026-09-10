HOME_DIR = "/mnt/storage/franklhtan/projects/TalkerVar2026"

REC_ROOT = (
    "/mnt/storage/franklhtan/projects/"
    "TalkerVar2026/data/rec"
)

INCLUDED_SPEAKERS = None

INPUT_FILENAME = "metadata_filtered_vowels.csv"
OUTPUT_FILENAME = "metadata_vowels_acoustics.csv"

# ============================================================
# Acoustic extraction
# ============================================================
PITCH_FLOOR = 60.0
PITCH_CEILING = 500.0

MAXIMUM_FORMANT = 5500.0
FORMANT_WINDOW_LENGTH = 0.025

# Measure formants over central 40% of vowel
FORMANT_MEASURE_START = 0.30
FORMANT_MEASURE_END = 0.70

# Sample every 5 ms within that region
FORMANT_SAMPLE_STEP = 0.005

# include field
INCLUDE_FIELDS = ["label", "speaker_id", "sentence_id", "segment_id", "duration_s", "f0_median_hz", "f0_mean_hz", "f1_hz", "f2_hz", "f3_hz"]