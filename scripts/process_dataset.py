import pandas as pd 

from src.data_loader import load_spam_data
from src.preprocessor import TextPreprocessor
from src.config import DATA_PROCESSED_PATH, DATA_RAW_PATH


def safe_process(text, processor):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    return processor.process(text)


def main():
    df = load_spam_data(DATA_RAW_PATH)
    print(f"Загружено {len(df)} сообщений")

    processor = TextPreprocessor()

    df['clean_message'] = df['message'].apply(lambda x: safe_process(x, processor))

    output_path = DATA_PROCESSED_PATH / "spam_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()



    