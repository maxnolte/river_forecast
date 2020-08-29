import requests
import pandas as pd
import os


def convert_to_hourly_flow(flow_df, days=3, include_first_value=True):

    """ Convert the raw 30 min frequency flow from rivermap.ch into a
        padded 1hr frequency series with missing values filled in backwards.

    :param flow_df: Rivermap.ch flow values with missing values.
    :param days: The number of padded days to return
    :param include_first_value: Whether to include the first value (e.e. 1 day gives 25 values instead of 24)
    :return: The shortened and filled in river flow (no na values).
    """

    filled_flow_df = flow_df.asfreq('30min', method='bfill')
    last_timestamp = filled_flow_df.index[-1]
    if include_first_value:
        first_time_stamp = last_timestamp - pd.Timedelta(days=days)
    else:
        first_time_stamp = last_timestamp - pd.Timedelta(days=days, hours=-1)
    idx = pd.date_range(start=first_time_stamp, end=last_timestamp, freq='H')
    hourly_filled_flow_df = filled_flow_df.reindex(idx).fillna(method='bfill')
    return hourly_filled_flow_df


class RivermapDataRetriever:

    """
    This is a class for retrieving data from the rivermap.ch API.
    A private API key is required, contact 'Rivermap supporters' on Facebook, or contact rivermap.ch.

    This class only works for the Dranse so far, but can be extended easily.
    """

    station_id_dict = {'Dranse': 'b49e45e5-73ce-3e0d-ade8-945e713307c5'}

    def __init__(self):
        self.api_key = os.environ.get("PRIVATE_RIVERZONE_API_KEY")

    def get_latest_river_flow(self, n_days=5, station='Dranse'):

        """
        Return the latest river discharge values from rivermap.ch API

        :param n_days: Number of past days to include
        :param station: Number of station
        :return: DataFrame with time and discharge
        """

        station_id = self.station_id_dict[station]
        url = f"https://api.riverzone.eu/v2/stations/{station_id}/readings"
        params = {'key': self.api_key,
                  'from': 60 * 24 * n_days, 'to': 60 * 24 * n_days}
        with requests.get(url=url, params=params) as req:
            readings_json = req.json()
        flow_df = pd.DataFrame(readings_json['readings']['m3s'])
        flow_df = flow_df.rename(columns={'ts': 'datetime', 'v': 'discharge'})
        flow_df['datetime'] = flow_df['datetime'].apply(pd.to_datetime, origin='unix', unit='s', utc=True)
        flow_df = flow_df.set_index('datetime')
        # Remove time zone information - keep time local (French)
        flow_df.index = flow_df.index.tz_convert('Europe/Paris').tz_localize(None)
        return flow_df

    def get_standard_dranse_data(self):

        """
        Retrive flow data for Dranse and format and pad for forecast.

        :return: Formatted recent Dranse flow.
        """

        flow_df = self.get_latest_river_flow(n_days=4, station='Dranse')
        return convert_to_hourly_flow(flow_df, days=3, include_first_value=False)
