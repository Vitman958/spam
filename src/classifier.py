import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from src.preprocessor import TextPreprocessor


class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.90, sublinear_tf=True)
        self.model = MultinomialNB()
        self.is_trained = False
        self.preprocessor = TextPreprocessor()

    def train(self, df: pd.DataFrame, text_col: str = 'clean_message', label_col: str = 'label'):
        x = df[text_col]
        y = df[label_col].map({'ham': 0, 'spam': 1})
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )

        x_train_vec = self.vectorizer.fit_transform(x_train)
        x_test_vec = self.vectorizer.transform(x_test)

        self.model.fit(x_train_vec, y_train)

        print("Отчёт по качеству:")
        print(classification_report(y_test, self.model.predict(x_test_vec)))

        self.is_trained = True
        return x_test, y_test
    
    def predict(self, text: str) -> dict:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        clean_text = self.preprocessor.process(text)

        text_vec = self.vectorizer.transform([clean_text])
        proba = self.model.predict_proba(text_vec)[0]
        prob_spam = proba[1]

        threshold = 0.75
        return {
        'is_spam': bool(prob_spam > threshold),
        'probability': float(prob_spam)
        }

    def evaluate(self, x_test, y_test):
        if not self.is_trained:
            raise ValueError('Модель не обучена')
        
        x_test_vec = self.vectorizer.transform(x_test)

        y_pred = self.model.predict(x_test_vec)

        print("Оценка на предоставленных данных:")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))        
    
    def save(self, path: str):
        joblib.dump(self, path)

    def load(self, path: str):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        self.is_trained = True