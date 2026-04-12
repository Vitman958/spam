import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW_PATH = BASE_DIR / "data"/ "raw" / "spam.csv"
DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NLTK_DATA_PATH = BASE_DIR / "nltk_data" 