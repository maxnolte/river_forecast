import pickle
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')


class Forecast:


    def generate_prediction_plot(self, recent_flow, n_hours=6, show=False):
        """

        :param recent_flow:
        :param n_hours:
        :return:
        """
        n_last_hours = 24

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(10, 3))
        forecast_flow = self.dynamic_forecast(recent_flow, n_hours=n_hours)
        recent_flow = recent_flow.iloc[-n_last_hours:]
        ax.plot(pd.concat([recent_flow['discharge'].iloc[-1:], forecast_flow.iloc[:1]]), marker='',
                linestyle='dashed', color='#7fcdbb')
        ax.plot(forecast_flow, marker='.', linestyle='dashed', color='#7fcdbb');
        ax.plot(recent_flow, marker='.', color='#2c7fb8');
        ax.set_ylabel('River flow (m3/s)')

        ax.set_xticks(pd.concat([recent_flow['discharge'], forecast_flow]).index)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a, %H:%M"))
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        _ = plt.xticks(rotation=45)
        last_time = recent_flow.iloc[-1:].index.strftime("%c")[0]
        ax.set_title(f'{self.__class__.__name__} ({last_time})')
        if show:
            plt.show()
        return fig, ax

    def dynamic_forecast(self, recent_flow, n_hours=6):
        pass


class NaiveForecast(Forecast):

    def dynamic_forecast(self, recent_flow, n_hours=6):
        """
        :param recent_flow:
        :param n_hours:
        :return:
        """

        last_timestamp = recent_flow.iloc[-1:].index.to_pydatetime()[0]
        forecast_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1),
                                       last_timestamp + pd.Timedelta(hours=n_hours), freq="h")
        return pd.Series([recent_flow.iloc[-1].values[0] for _ in range(n_hours)], index=forecast_index)


class SARIMAXForecast(Forecast):

    model_fit_recent = None

    def __init__(self, model_params_path='../models/sarimax_411_011-24_model-parameters.pkl'):
        file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model_params = self.load_SARIMAX_params(file_path)

    def load_SARIMAX_params(self, model_params_path):
        return pickle.load(open(model_params_path, 'rb'))

    def update_forecast_SARIMAX(self, recent_flow):
        model = SARIMAX(recent_flow, order=(4, 1, 1), seasonal_order=(0, 1, 1, 24))
        self.model_fit_recent = model.smooth(self.model_params)

    def dynamic_forecast(self, recent_flow, n_hours=6):
        self.update_forecast_SARIMAX(recent_flow)
        return self.model_fit_recent.forecast(steps=n_hours)


class LSTMForecast(Forecast):

    def __init__(self, model_params_path='../models/LSTM_model'):
        file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model = tf.keras.models.load_model(file_path)

    def dynamic_forecast(self, recent_flow, n_hours=6):

        recent_flow_diff = recent_flow.diff(periods=1).dropna()['discharge']

        pred_flow = self.multi_step_forecast_from_diff(recent_flow_diff,
                                                       recent_flow.iloc[-1]['discharge'],
                                                       n_steps=n_hours)

        last_timestamp = recent_flow.iloc[-1:].index.to_pydatetime()[0]
        forecast_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1),
                                       last_timestamp + pd.Timedelta(hours=n_hours), freq="h")
        return pd.Series(pred_flow, index=forecast_index)

    def multi_step_forecast(self, time_series, n_steps=12):
        time_series = np.array(time_series).reshape(1, time_series.size, 1)
        forecast = np.zeros(n_steps, dtype=float)
        for i in range(n_steps):
            time_series = self.model.predict(time_series)
            forecast[i] = time_series[0, -1, 0]
        return forecast

    def multi_step_forecast_from_diff(self, time_series, last_value, n_steps=12):
        forecast = self.multi_step_forecast(time_series, n_steps=n_steps)
        f_t = last_value
        for i in range(n_steps):
            forecast[i] = f_t + forecast[i]
            f_t = forecast[i]
        return forecast


class LSTMSeq2SeqForecast(Forecast):

    def __init__(self, model_params_path='../models/LSTM_model_v2'):
        file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model = tf.keras.models.load_model(file_path, custom_objects={'tf': tf})

    def dynamic_forecast(self, recent_flow, n_hours=6, n_hours_in=24):
        recent_flow = recent_flow.iloc[-n_hours_in:]

        pred_flow = self.multi_step_forecast_seq2seq(recent_flow).reshape(-1)
        last_timestamp = recent_flow.iloc[-1:].index.to_pydatetime()[0]
        forecast_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1),
                                       last_timestamp + pd.Timedelta(hours=n_hours), freq="h")
        return pd.Series(pred_flow, index=forecast_index)

    def multi_step_forecast_seq2seq(self, time_series):
        time_series = np.array(time_series).reshape(1, time_series.size, 1)
        forecast = self.model.predict(time_series)
        return forecast


class XGBForecast(Forecast):
    """

    """

    def __init__(self, model_params_path='../models/XGB_models_v1.pkl'):
        file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model_dict = pickle.load(open(file_path, 'rb'))

    def dynamic_forecast(self, recent_flow, n_hours=6, n_hours_in=48):
        breakpoint()
        x_pred = self.create_features(recent_flow, last_n_steps=48).iloc[-1:]
        pred_flow = self.get_predictions_from_model_dict(x_pred)
        last_timestamp = recent_flow.iloc[-1:].index.to_pydatetime()[0]
        forecast_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1),
                                       last_timestamp + pd.Timedelta(hours=n_hours), freq="h")
        return pd.Series(pred_flow, index=forecast_index)

    def get_predictions_from_model_dict(self, validation_x):
        y = []
        for name, model in self.model_dict.items():
            y.append(model.predict(validation_x))
        return y

    def create_features(self, flow_df, last_n_steps=6):
        df = flow_df.copy()
        for i in range(1, last_n_steps):
            df[f'discharge_{i}'] = df['discharge'].shift(periods=i)

        for i in range(0, last_n_steps):
            df[f'discharge_diff_{i}'] = df['discharge'].diff()

        for i in range(0, last_n_steps):
            df[f'discharge_diff_24_{i}'] = df['discharge'].diff(periods=24)

        df['hour'] = df.index.hour.values
        df = df.dropna()
        return df
