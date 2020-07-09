import pickle
import seaborn as sns
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('TkAgg')

class Forecast:

    def __init__(self):
        pass

    def generate_prediction_plot(self, recent_flow, n_hours=6, show=False):
        """

        :param recent_flow:
        :param n_hours:
        :return:
        """
        n_last_hours = 24

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(10, 3))
        forecast_flow = self.dynamic_forecast(recent_flow)
        recent_flow = recent_flow.iloc[-n_last_hours:]
        ax.plot(pd.concat([recent_flow['discharge'].iloc[-1:], forecast_flow.iloc[:1]]), marker='',
                linestyle='dashed', color='#7fcdbb');
        ax.plot(forecast_flow, marker='.', linestyle='dashed', color='#7fcdbb');
        ax.plot(recent_flow, marker='.', color='#2c7fb8');
        ax.set_ylabel('River flow (m3/s)')

        ax.set_xticks(pd.concat([recent_flow['discharge'], forecast_flow]).index)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a, %H:%M"))
        #ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        _ = plt.xticks(rotation=45)
        last_time = recent_flow.iloc[-1:].index.strftime("%c")[0]
        ax.set_title(f'SARIMA Forecast ({last_time})')
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

    def __init__(self, model_params_path='../models/sarimax_211_011-24_model-parameters.pkl'):
        self.model_params = self.load_SARIMAX_params(model_params_path)

    def load_SARIMAX_params(self, model_params_path):
        return pickle.load(open(model_params_path, 'rb'))

    def update_forecast_SARIMAX(self, recent_flow):
        model = SARIMAX(recent_flow, order=(4, 1, 1), seasonal_order=(0, 1, 1, 24))
        self.model_fit_recent = model.smooth(self.model_params)

    def dynamic_forecast(self, recent_flow, n_hours=6):
        self.update_forecast_SARIMAX(recent_flow)
        return self.model_fit_recent.forecast(steps=n_hours)
