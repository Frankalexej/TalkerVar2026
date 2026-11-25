
# ============================================================
# Paths
# ============================================================

HOME_DIR = "/mnt/storage/franklhtan/projects/TalkerVar2026"

# Replace with the actual original JVS dataset location
JVS_ROOT = (
    "/mnt/storage/franklhtan/projects/"
    "TalkerVar2026/data/jvs_ver1"
)

INPUT_FILENAME = "metadata_filtered_vowels.csv"
OUTPUT_FILENAME = "metadata_vowels_acoustics_sentence.csv"


# ============================================================
# Speaker selection
# ============================================================

INCLUDED_SPEAKERS = None


# ============================================================
# Pitch analysis
# ============================================================

PITCH_FLOOR = 60.0
PITCH_CEILING = 500.0
PITCH_TIME_STEP = 0.005


# ============================================================
# Formant analysis
# ============================================================

MAXIMUM_FORMANT = 5500.0
FORMANT_WINDOW_LENGTH = 0.025
FORMANT_TIME_STEP = 0.005

FORMANT_MEASURE_START = 0.30
FORMANT_MEASURE_END = 0.70


# ============================================================
# Quality control
# ============================================================

BOUNDARY_TOLERANCE_S = 0.001
SAVE_ACOUSTIC_QC = True


# ============================================================
# Output fields
# ============================================================

INCLUDE_FIELDS = [
    "label",
    "speaker_id",
    "sentence_id",
    "segment_id",
    "duration_s",
    "f0_median_hz",
    "f0_mean_hz",
    "f1_hz",
    "f2_hz",
    "f3_hz",
]
