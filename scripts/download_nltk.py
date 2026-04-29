import nltk
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NLTK_DATA_PATH  


def download_nltk_data() -> bool:
    """
    Скачивает необходимые корпуса NLTK в папку проекта.
    """
    try:
        NLTK_DATA_PATH.mkdir(exist_ok=True)
        
        resources = [
            'stopwords',
            'wordnet', 
            'omw-1.4',
            'punkt',
            'punkt_tab'
        ]
        
        for resource in resources:
            print(f"Скачиваю {resource}...")
            success = nltk.download(resource, download_dir=NLTK_DATA_PATH)
            
            if not success:
                print(f"Ошибка при скачивании {resource}")
                return False
        
        print(f"Все корпуса загружены в {NLTK_DATA_PATH}")
        return True
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return False


if __name__ == "__main__":
    success = download_nltk_data()
    