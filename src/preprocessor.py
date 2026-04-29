import string
import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import NLTK_DATA_PATH


nltk.data.path.append(str(NLTK_DATA_PATH))


class TextPreprocessor:
    def __init__(self, custom_stop_words: List[str] = None):
        self.lemmatizer = WordNetLemmatizer()
        base_stop_words = set(stopwords.words('english'))
        
        if custom_stop_words:
            base_stop_words.update(custom_stop_words)

        self.stop_words = base_stop_words

    def lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        trans_table = str.maketrans('', '', string.punctuation)
        return text.translate(trans_table)
    
    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text: str) -> str:
        text = self.lowercase(text)
        
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
        text = re.sub(r'<[^>]+>', '<HTML>', text)
        text = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', text)
        text = re.sub(r'\$|€|£', '<MONEY>', text)
        text = re.sub(r'!{1,}', '<ALERT>', text)
        text = re.sub(r'\s+', ' ', text).strip()             
        
        text = text.replace('-', ' ')
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)


proc = TextPreprocessor()
