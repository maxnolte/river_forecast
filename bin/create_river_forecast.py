from river_forecast.api_data_access import RivermapDataRetriever
from river_forecast.forecast import SARIMAXForecast

recent_data = RivermapDataRetriever().get_standard_dranse_data()
SARIMAXForecast().generate_prediction_plot(recent_data, show=True)