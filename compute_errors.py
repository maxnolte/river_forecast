from river_forecast.api_data_access import RivermapDataRetriever
from river_forecast.forecast import SARIMAXForecast, NaiveForecast, LSTMForecast, LSTMSeq2SeqForecast, XGBForecast

# recent_data = RivermapDataRetriever().get_standard_dranse_data()
# SARIMAXForecast().generate_prediction_plot(recent_data, show=True)
# NaiveForecast().generate_prediction_plot(recent_data, show=True)
# LSTMForecast().generate_prediction_plot(recent_data, show=True)
# LSTMSeq2SeqForecast().generate_prediction_plot(recent_data, show=True)

xgb = XGBForecast()

forecasts, real_values = xgb.compute_real_and_predicted_values_test_data(n_forecasts=20)

print(forecasts.shape, real_values.shape)


forecasts, real_values = xgb.get_real_and_predicted_values_test_data(recompute=False)

print(forecasts.shape, real_values.shape)


error_metrics = xgb.get_error_metrics(recompute=False, ci=60)

print(error_metrics)