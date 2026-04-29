import pandas as pd

from src.classifier import SpamClassifier
from src.config import DATA_PROCESSED_PATH, MODELS_DIR


def main():
    df = pd.read_csv(DATA_PROCESSED_PATH / "email_clean.csv")

    classifier = SpamClassifier()
    classifier.train(df)
    
    classifier.save(MODELS_DIR / "spam_model.joblib")

if __name__ == "__main__":
    main()