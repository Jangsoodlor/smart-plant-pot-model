import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ModelManager:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        predicting_column: str,
        order: tuple[int, int, int],
        time_delta: pd.Timedelta,
    ):
        """
        Init the model
        :param dataframe: The dataframe containing all the columns
        :param predicting_column: The name of the predicting column
        :param exog_columns: The list of exog columns.
        :param order: The order of ARIMAX model
        """
        self.__main_data = dataframe
        self.__predicting_column = predicting_column
        self.__exog_models = []
        self.__exog_columns = []
        self.__exog_training_values = None
        self.__exog_predicting_values = None
        self.__order = order
        self.__model = None
        self.__timedelta = time_delta

    def add_exog(
        self,
        column_name: str,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ):
        self.__exog_columns.append(column_name)
        model = SARIMAX(
            endog=self.__main_data[column_name],
            order=order,
            seasonal_order=seasonal_order,
        )
        self.__exog_models.append({"name": column_name, "model": model.fit()})
        return self

    def __get_exog_data(self, steps: int):
        for model in self.__exog_models:
            predicted = model["model"].get_forecast(steps=steps).predicted_mean
            predicted.columns = [model["name"]]
            if self.__exog_predicting_values is None:
                self.__exog_predicting_values = predicted
            else:
                self.__exog_predicting_values = pd.concat(
                    [self.__exog_predicting_values, predicted], axis=1
                )

    def fit_model(self):
        if self.__exog_columns[0] is None:
            model = SARIMAX(
                endog=self.__main_data[self.__predicting_column], order=self.__order
            )
        else:
            model = SARIMAX(
                endog=self.__main_data[self.__predicting_column],
                exog=self.__main_data[self.__exog_columns],
                order=self.__order,
            )
        self.__model = model.fit()
        return self

    def get_prediction(self, steps: int):
        if self.__model is None:
            raise AttributeError("The model is not fitted yet")
        self.__get_exog_data(steps)
        predicted_obj = self.__model.get_forecast(
            steps=steps, exog=self.__exog_predicting_values
        )
        predicted = predicted_obj.predicted_mean
        forecast_index = pd.date_range(
            start=self.__main_data.index[-1] + self.__timedelta,
            freq=self.__timedelta,
            periods=steps,
        )
        predicted.columns = ["predicted"]
        predicted.index = forecast_index
        forecast_ci = predicted_obj.conf_int()
        lower = forecast_ci.iloc[:, 0]
        lower.index = forecast_index
        upper = forecast_ci.iloc[:, 1]
        upper.index = forecast_index
        print(predicted)
        print("date range: ", forecast_index)

        return predicted, upper, lower

    def __new__(cls):
        """Singleton __new__ constructor"""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", index_col="ts", parse_dates=True)
    print(df.columns)
    model_manager = ModelManager(
        df, "soil_moisture", (2, 0, 1), pd.Timedelta(minutes=15)
    )
    prediction, upper, lower = (
        model_manager.add_exog("temperature", (2, 0, 1), (1, 0, 1, 24))
        .add_exog("humidity", (2, 0, 1), (1, 0, 1, 24))
        .fit_model()
        .get_prediction(1240)
    )
    plt.figure(figsize=(12, 6))
    # Ensure datetime index and proper plotting
    plt.plot(df.index, df["soil_moisture"], label="data")
    print(df.index, df["soil_moisture"])
    print("-----------------------------------")
    print(prediction.index, prediction)
    plt.plot(
        prediction.index, prediction, label="Predicted"
    )  # use the same x-axis as test
    plt.fill_between(prediction.index, lower, upper, color="pink", alpha=0.3)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture")
    plt.title("Soil Moisture: Training, Test and Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
