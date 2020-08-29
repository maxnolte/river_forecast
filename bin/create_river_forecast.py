from river_forecast.api_data_access import RivermapDataRetriever
from river_forecast.forecast import SARIMAXForecast, NaiveForecast, LSTMForecast, LSTMSeq2SeqForecast, XGBForecast

recent_data = RivermapDataRetriever().get_standard_dranse_data()
# SARIMAXForecast().generate_prediction_plot(recent_data, show=True)
# NaiveForecast().generate_prediction_plot(recent_data, show=True)
# LSTMForecast().generate_prediction_plot(recent_data, show=True)
# LSTMSeq2SeqForecast().generate_prediction_plot(recent_data, show=True)

XGBForecast().generate_prediction_plot(recent_data, show=True, ci=True, save_png=True)


