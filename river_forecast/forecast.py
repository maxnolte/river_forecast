import pickle
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


class Forecast:

    def __init__(self):
        pass

    def generate_prediction_plot(self, recent_flow, n_hours=6):
        """

        :param recent_flow:
        :param n_hours:
        :return:
        """
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(recent_flow, marker='.');
        ax.plot(self.dynamic_forecast(recent_flow), marker='.');
        return fig, ax

    def dynamic_forecast(self, recent_flow, n_hours=6):
        pass


class SARIMAXForecast(Forecast):

    model_fit_recent = None

    def __init__(self, model_path='models/sarimax_411_011-24_model.pkl'):
        self.model_fit = self.load_SARIMAX(model_path)

    def load_SARIMAX(self, model_path):
        return pickle.load(open(model_path, 'rb'))

    def update_forecast_SARIMAX(self, recent_flow):
        model = SARIMAX(recent_flow, order=(4, 1, 1), seasonal_order=(0, 1, 1, 24))
        self.model_fit_recent = model.smooth(self.model_fit.params)

    def dynamic_forecast(self, recent_flow, n_hours=6):
        self.update_forecast_SARIMAX(recent_flow)
        return self.model_fit_recent.forecast(steps=n_hours)
