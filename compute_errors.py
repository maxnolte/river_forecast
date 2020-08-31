from river_forecast.api_data_access import RivermapDataRetriever
from river_forecast.forecast import SARIMAXForecast, NaiveForecast, LSTMForecast, LSTMSeq2SeqForecast, XGBForecast


models = [SARIMAXForecast(), NaiveForecast(), LSTMForecast(), LSTMSeq2SeqForecast()]

for model in models:
    model.get_real_and_predicted_values_test_data(recompute=False)
