from river_forecast.api_data_access import HubeauDataRetriever
from river_forecast.forecast import XGBForecast

recent_data = HubeauDataRetriever().get_standard_dranse_data()
XGBForecast().generate_prediction_plot(recent_data, show=False, ci=True, save_png=True)
