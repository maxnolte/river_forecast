import requests
import pandas as pd
import os


class RivermapDataRetriever:

    station_id_dict = {'Dranse': 'b49e45e5-73ce-3e0d-ade8-945e713307c5'}

    def __init__(self):
        self.api_key = os.environ.get("PRIVATE_RIVERZONE_API_KEY")

    def get_latest_river_flow(self, n_days=5, station='Dranse'):
        """
        Return the latest river discharge values from rivermap.ch API
        :param n_days: Number of past days to include
        :param station: Nambe of station
        :return: DataFrame with time and discharge
        """
        # URL for Dranse @ Bioge
        station_id = self.station_id_dict[station]
        url = f"https://api.riverzone.eu/v2/stations/{station_id}/readings"
        params = {'key': self.api_key,
                  'from': 60 * 24 * n_days, 'to': 60 * 24 * n_days}
        with requests.get(url=url, params=params) as req:
            readings_json = req.json()
        flow_df = pd.DataFrame(readings_json['readings']['m3s'])
        flow_df = flow_df.rename(columns={'ts': 'datetime', 'v': 'discharge'})
        flow_df['datetime'] = flow_df['datetime'].apply(pd.to_datetime, origin='unix', unit='s')
        flow_df = flow_df.set_index('datetime')
        return flow_df

    def convert_to_hourly_flow(self, flow_df, days=3, include_first_value=True):
        """

        :param flow_df:
        :return:
        """
        filled_flow_df = flow_df.asfreq('30min', method='bfill')
        last_timestamp = filled_flow_df.index[-1]
        if include_first_value:
            first_time_stamp = last_timestamp - pd.Timedelta(days=days)
        else:
            first_time_stamp = last_timestamp - pd.Timedelta(days=days, hours=-1)
        hourly_filled_flow_df = filled_flow_df.loc[first_time_stamp:last_timestamp].asfreq('H')
        return hourly_filled_flow_df

    def get_standard_dranse_data(self):
        flow_df = self.get_latest_river_flow(n_days=4, station='Dranse')
        return self.convert_to_hourly_flow(flow_df, days=3, include_first_value=False)
