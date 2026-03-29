from pathlib import Path

import pandas as pd
from config import DATA_RAW_PATH


def load_spam_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='ISO-8859-1')
    
    df.rename(columns={'v1': 'label', 'v2': "message"}, inplace=True)

    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding='ISO-8859-1')

