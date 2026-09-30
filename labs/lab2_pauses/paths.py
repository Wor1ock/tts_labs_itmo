"""Пути к данным. Считаются от расположения этого файла, а не от текущей директории."""
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent

# --- данные, общие для всех лаб ---
ROOT_DATA_DIR = LAB_DIR.parents[1] / 'data'
RUSLAN_METADATA_PATH = ROOT_DATA_DIR / 'metadata_RUSLAN_22200_normalized.csv'
RUSLAN_ALIGN_DIR = ROOT_DATA_DIR / 'RUSLAN_align' / 'v2'
RUSLAN_AUDIO_DIR = ROOT_DATA_DIR / 'RUSLAN'

# --- данные текущей лабы ---
LAB_DATA_DIR = LAB_DIR / 'data'
PAUSE_METADATA_PATH = LAB_DATA_DIR / 'RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = LAB_DATA_DIR / 'RUSLAN_pause_features.csv'
AUDIO_REVIEW_HTML = LAB_DATA_DIR / 'audio_review.html'
