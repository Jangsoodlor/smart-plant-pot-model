import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ModelManager:
    __instance = None

    @classmethod
    def get_model(cls):
        return cls.__instance

    @classmethod
    def set_model(cls, instance):
        cls.__instance = instance


class ModelBuilder:
    def __init__(self):
        self.main_data = None
        self.exog_models = []
        self.exog_columns = []
        self.predicting_column = None
        self.exog_training_values = None
        self.exog_predicting_values = None
        self.order = None
        self.model = None
        self.timedelta = None

    def add_exog(
        self,
        column_name: str,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ):
        self.exog_columns.append(column_name)
        model = SARIMAX(
            endog=self.main_data[column_name],
            order=order,
            seasonal_order=seasonal_order,
        )
        self.exog_models.append({"name": column_name, "model": model.fit()})
        return self

    def add_basic_init(
        self,
        dataframe: pd.DataFrame,
        predicting_column: str,
        order: tuple[int, int, int],
        time_delta: pd.Timedelta,
    ):
        self.main_data = dataframe
        self.predicting_column = predicting_column
        self.order = order
        self.timedelta = time_delta
        return self

    def build(self):
        model = Model(self)
        ModelManager.set_model(model)
        return model


class Model:
    def __init__(self, builder: ModelBuilder):
        """
        Init the model
        :param dataframe: The dataframe containing all the columns
        :param predicting_column: The name of the predicting column
        :param exog_columns: The list of exog columns.
        :param order: The order of ARIMAX model
        """
        self.__main_data = builder.main_data
        self.__predicting_column = builder.predicting_column
        self.__exog_models = builder.exog_models
        self.__exog_columns = builder.exog_columns
        self.__exog_training_values = builder.exog_training_values
        self.__exog_predicting_values = builder.exog_predicting_values
        self.__order = builder.order
        self.__model = builder.model
        self.__timedelta = builder.timedelta
        self.__cached_prediction = []  # [prediction, upper, lower] #TODO verify that this works

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
        if len(self.__cached_prediction) == 3:
            return (
                self.__cached_prediction[0],
                self.__cached_prediction[1],
                self.__cached_prediction[2],
            )
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

        self.__cached_prediction = [predicted, upper, lower]

        return predicted, upper, lower

    def get_duration(self, moisture_level) -> str:
        """Returns either the duration based on moisture level (or the exact date and time up 2 you)"""
        # TODO: implement
        return "3000 days"


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", index_col="ts", parse_dates=True)
    builder = ModelBuilder()
    builder.add_basic_init(df, "soil_moisture", (2, 0, 1), pd.Timedelta(minutes=15))

    builder.add_exog("temperature", (2, 0, 1), (1, 0, 1, 24)).add_exog(
        "humidity", (2, 0, 1), (1, 0, 1, 24)
    ).build()
    prediction, upper, lower = ModelManager.get_model().fit_model().get_prediction(1240)

    fig = go.Figure()

    # Line for actual data
    fig.add_trace(
        go.Scatter(x=df.index, y=df["soil_moisture"], mode="lines", name="data")
    )

    # Line for predicted data
    fig.add_trace(
        go.Scatter(x=prediction.index, y=prediction, mode="lines", name="Predicted")
    )

    # Fill between lower and upper prediction intervals
    fig.add_trace(
        go.Scatter(
            x=prediction.index.tolist() + prediction.index[::-1].tolist(),
            y=upper.tolist() + lower[::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255, 192, 203, 0.3)",  # pink with alpha 0.3
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Interval",
        )
    )

    fig.update_layout(
        title="Soil Moisture: Training, Test and Predictions",
        xaxis_title="Date",
        yaxis_title="Soil Moisture",
        legend=dict(title=None),
        template="simple_white",
    )

    fig.show()
