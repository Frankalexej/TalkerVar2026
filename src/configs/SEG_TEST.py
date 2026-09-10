from pathlib import Path

PROJECT_ROOT = Path("/mnt/storage/franklhtan/projects/TalkerVar2026")

JVS_ROOT = PROJECT_ROOT / "data" / "jvs_ver1"
REC_ROOT = PROJECT_ROOT / "data" / "rec"
LAB_ROOT = PROJECT_ROOT / "data" / "lab"

DATA_TYPES = {"parallel100", "nonpara30"}

# Empty = process all speakers
# SPEAKER_DIRS = set()

# Example: process only selected speakers
SPEAKER_DIRS = {
    "jvs001"
}