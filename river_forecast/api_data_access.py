import requests
import pandas as pd
import os

def get_latest_river_flow(n_days=5, station_id='b49e45e5-73ce-3e0d-ade8-945e713307c5'):
    # URL for Dranse @ Bioge
    url = f"https://api.riverzone.eu/v2/stations/{station_id}/readings"
    api_key = os.environ.get("PRIVATE_RIVERZONE_API_KEY")
    params = {'key': api_key,
              'from': 60 * 24 * n_days, 'to': 60 * 24 * n_days}
    with requests.get(url=url, params=params) as req:
        readings_json = req.json()
    flow_df = pd.DataFrame(readings_json['readings']['m3s'])
    flow_df['ts'] = flow_df['ts'].apply(pd.to_datetime, origin='unix', unit='s')
    return flow_df

