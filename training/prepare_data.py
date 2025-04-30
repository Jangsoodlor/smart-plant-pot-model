from datetime import datetime, timedelta

import pandas as pd
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)


class PrepareData:
    """Prepares the data by resampling and also split train-test."""
    __instance = None
    df = (
        pd.read_csv("../dataset.csv", index_col="ts", parse_dates=True)
        .resample("6H")
        .mean()
    )
    # Using KNN Imputer to replace the missing values, this should work better than replacing with previous value.
    indexes = df.index
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df.index = indexes
    train_end = datetime(2025, 4, 9)
    test_end = datetime(2025, 4, 11)

    @classmethod
    def train_data(cls) -> pd.DataFrame:
        return cls.df[: cls.train_end - timedelta(hours=6)]

    @classmethod
    def test_data(cls) -> pd.DataFrame:
        return cls.df[cls.train_end : cls.test_end]
    
    
if __name__ == "__main__":

    test = PrepareData.test_data()
    print(test)
