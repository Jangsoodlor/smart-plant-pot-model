from datetime import datetime, timedelta

import pandas as pd


class PrepareData:
    """Prepares the data by resampling and also split train-test."""
    __instance = None
    df = (
        pd.read_csv("../dataset.csv", index_col="ts", parse_dates=True)
        .resample("6H")
        .mean()
    )
    train_end = datetime(2025, 4, 9)
    test_end = datetime(2025, 4, 11)

    @classmethod
    def train_data(cls) -> pd.DataFrame:
        return cls.df[: cls.train_end - timedelta(days=1)]

    @classmethod
    def test_data(cls) -> pd.DataFrame:
        return cls.df[cls.train_end : cls.test_end]
