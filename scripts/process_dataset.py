import pandas as pd 
from pathlib import Path

from src.preprocessor import TextPreprocessor
from src.config import DATA_PROCESSED_PATH, DATA_RAW_PATH


def safe_process(text, processor):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    return processor.process(text)


def main():
    df = pd.read_csv(DATA_RAW_PATH)
    print(f"Загружено {len(df)} строк")

    df.rename(columns={'text': 'message', 'label': 'raw_label'}, inplace=True)
    df['label'] = df['raw_label'].map({0: 'ham', 1: 'spam'})
    
    df = df[['label', 'message']].copy()

    df = df.dropna(subset=['message'])
    df = df[df['message'].str.strip() != '']
    print(f"Валидных сообщений: {len(df)}")

    processor = TextPreprocessor()
    print("Обработка текст")
    df['clean_message'] = df['message'].apply(lambda x: safe_process(x, processor))

    df = df[df['clean_message'].str.strip() != ''].copy()

    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_PATH / "email_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"Сохранено в {output_path}")
    print(f"Распределение классов:\n{df['label'].value_counts()}")
    print(f"Средняя длина очищенного текста: {df['clean_message'].str.len().mean():.0f} символов")


if __name__ == "__main__":
    main()