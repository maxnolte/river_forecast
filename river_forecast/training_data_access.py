import pandas as pd
from pathlib import Path

def get_yearly_flow(year=2016):
    """ Reading text files with yearly data.
    """
    file_path = (Path(__file__).parent / f"../data/hourly_flows_2016-19/flows-{year}.txt").resolve()
    flow_df = pd.read_csv(file_path, delimiter='\t').drop(columns='V')
    flow_df.columns = ['datetime', 'discharge']
    flow_df['datetime'] = pd.to_datetime(flow_df['datetime'], dayfirst=True)
    flow_df = flow_df.set_index('datetime')
    return flow_df

def get_combined_flow():
    """ Combining dataframes from individual years into one dataframe.
    """
    dfs = []
    for year in range(2016, 2021):
        dfs.append(get_yearly_flow(year=year))
    flow_df = pd.concat(dfs)
    flow_df = flow_df.asfreq('H').interpolate()
    return flow_df

def get_combined_flow_split():

    flow_df = get_combined_flow()

    train = flow_df.loc[flow_df.index < pd.to_datetime('2018-07-01 00:00:00')]
    validation = flow_df.loc[(flow_df.index > pd.to_datetime('2018-07-01 00:00:00')) & (
                flow_df.index < pd.to_datetime('2019-07-01 00:00:00'))]
    test = flow_df.loc[flow_df.index > pd.to_datetime('2019-07-01 00:00:00')]

    return train, validation, test
