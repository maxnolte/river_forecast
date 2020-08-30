
import numpy as np
import pandas as pd

import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from river_forecast.training_data_access import get_combined_flow_split

# Needed to save plot on some systems
# matplotlib.use('TkAgg')


class Forecast:


    def generate_prediction_plot(self, recent_flow, n_hours=6, show=False, ci=False, save_png=False):
        """

        :param recent_flow:
        :param n_hours:
        :return:
        """
         
        n_last_hours = 24
        colors = ['.9', '#9ecae1', '#3182bd']

        sns.set_style('darkgrid')
        sns.set_style("darkgrid", {"axes.facecolor": colors[0]})

        fig, ax = plt.subplots(figsize=(10, 3))
        forecast_flow = self.dynamic_forecast(recent_flow, n_hours=n_hours)

        recent_flow = recent_flow.iloc[-n_last_hours:]
        forecast_flow_plot = pd.concat([recent_flow['discharge'].iloc[-1:], forecast_flow])


        ax.plot(recent_flow, marker='.', color=colors[2], label='Recent flow')
        ax.plot(forecast_flow_plot, marker='.', linestyle='dashed', color=colors[2], label='Predicted flow')

        if ci is True:
            confidence_interval = 80
            cis = self.get_error_metrics(ci=confidence_interval)['ci']

            negative_ci = np.insert(cis[0], 0, 1)
            positive_ci = np.insert(cis[1], 0, 1)

            ax.fill_between(forecast_flow_plot.index, forecast_flow_plot * negative_ci,
                           forecast_flow_plot * positive_ci, color=colors[1],
                            label=f'{confidence_interval}% confidence interval')

        ax.set_ylabel('River flow (m3/s)')

        ax.set_xticks(pd.concat([recent_flow['discharge'], forecast_flow]).index)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a, %H:%M"))
        _ = plt.xticks(rotation=45)

        last_time = recent_flow.iloc[-1:].index.strftime("%c")[0]
        ax.set_title(f'{self.__class__.__name__} ({last_time})')
        ax.legend(loc='upper left')
        if save_png:
            plt.savefig('forecast.png', dpi=300)
        if show:
            plt.show()
        return fig, ax


    def dynamic_forecast(self, recent_flow, n_hours=6):
        pass


    def get_error_metrics(self, recompute=False, ci=60):

        forecasts, real_values = self.get_real_and_predicted_values_test_data(recompute=recompute)

        relative_errors = forecasts / real_values

        margin = (100 - ci) / 2.0
        negative_ci = np.percentile(relative_errors, margin, axis=0)
        positive_ci = np.percentile(relative_errors, 100 - margin, axis=0)

        error_metrics = {'ci': [negative_ci, positive_ci],
                         'mape': np.mean(np.abs((forecasts - real_values) / real_values), axis=0),
                         'mae': np.mean(np.abs(forecasts - real_values), axis=0),
                         'rmse': np.sqrt(np.mean((forecasts - real_values) ** 2, axis=0))}

        return error_metrics


    def get_real_and_predicted_values_test_data(self, recompute=False):

        error_file_path = str(self.file_path) + '-error_distribution.npz'
        if os.path.isfile(error_file_path) and not recompute:
            data = np.load(error_file_path)
            forecasts = data['forecasts']
            real_values = data['real_values']
        else:
            forecasts, real_values = self.compute_real_and_predicted_values_test_data()
            np.savez(error_file_path, forecasts=forecasts, real_values=real_values)
        return forecasts, real_values



    def compute_real_and_predicted_values_test_data(self, n_forecasts=None):
        " Compute error distributions "

        train, validation, test = get_combined_flow_split()

        input_length = 72
        forecast_length = int(6)
        n_possible_forecasts = len(validation) - input_length + 1 - forecast_length

        if n_forecasts is None:
            n_forecasts = n_possible_forecasts

        forecasts = np.zeros((n_forecasts, forecast_length))
        real_values = np.zeros((n_forecasts, forecast_length))
        real_recent_flows = np.zeros((n_forecasts, 12))
        np.random.seed(5)
        for i, j in enumerate(np.random.choice(n_possible_forecasts, size=n_forecasts, replace=False)):
            if i % 50 == 0:
                print(i, 'out of', n_forecasts)
            recent_flow = validation.iloc[j:(j + input_length)]
            real_recent_flows[i, :] = recent_flow.iloc[-12:]['discharge']
            real_values[i, :] = validation.iloc[(j + input_length):(j + input_length + forecast_length)]['discharge']
            forecasts[i, :] = self.dynamic_forecast(recent_flow, n_hours=forecast_length)

        return forecasts, real_values


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
        self.file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model_params = self.load_SARIMAX_params(self.file_path)

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
        self.file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model = tf.keras.models.load_model(self.file_path)

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
        self.file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model = tf.keras.models.load_model(self.file_path, custom_objects={'tf': tf})

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
        self.file_path = (Path(__file__).parent / model_params_path).resolve()
        self.model_dict = pickle.load(open(self.file_path, 'rb'))

    def dynamic_forecast(self, recent_flow, n_hours=6, n_hours_in=48):
        x_pred = self.create_features(recent_flow, last_n_steps=48).iloc[-1:]
        pred_flow = self.get_predictions_from_model_dict(x_pred)
        last_timestamp = recent_flow.iloc[-1:].index.to_pydatetime()[0]
        forecast_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1),
                                       last_timestamp + pd.Timedelta(hours=n_hours), freq="h")
        return pd.Series(pred_flow, index=forecast_index)

    def get_predictions_from_model_dict(self, validation_x):
        y = []
        for name, model in self.model_dict.items():
            y.append(model.predict(validation_x)[0])
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
