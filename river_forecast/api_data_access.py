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

