import sys
import os
from pathlib import Path

import nltk


def get_resource_path(relative_path: str) -> Path:
    """Возвращает абсолютный путь к файлу"""

    if getattr(sys, 'frozen', False):
        
        base_path = Path(sys._MEIPASS)
    else:
        
        base_path = Path(__file__).resolve().parent.parent
    return base_path / relative_path


BASE_DIR = get_resource_path("")
DATA_RAW_PATH = get_resource_path("data/raw")
DATA_PROCESSED_PATH = get_resource_path("data/processed")
MODELS_DIR = get_resource_path("models")
NLTK_DATA_PATH = get_resource_path("nltk_data")

if str(NLTK_DATA_PATH) not in nltk.data.path:
    nltk.data.path.append(str(NLTK_DATA_PATH))