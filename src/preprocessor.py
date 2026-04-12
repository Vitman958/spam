import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import NLTK_DATA_PATH


nltk.data.path.append(str(NLTK_DATA_PATH))


class TextPreprocessor:
    def __init__(self, stop_words: List[str] = None):
        pass

    def lowercase(self, text: str) -> str:
        pass
    
    def remove_punctuation(self, text: str) -> str:
        # Подсказка: str.translate() + str.maketrans()
        pass
    
    def tokenize(self, text: str) -> List[str]:
        # Вернёт список слов
        pass
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        # Принимает список, возвращает список
        pass
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        # Принимает список, возвращает список
        pass
    
    def process(self, text: str) -> str:
        """Основной метод: цепочка всех преобразований"""
        # Реализуй пайплайн
        pass