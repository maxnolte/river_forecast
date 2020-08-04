import pandas as pd
from pathlib import Path

""" This module contains functions to access training, validation and testing data
for the river flow (discharge) of the Dranse river in Bioge, France.
    
Hourly values for the years 2016, 2017, 2018, and 2019.
"""


def get_yearly_flow(year=2016):

    """ Reading text files with yearly data.

       :return: A DataFrame with DateTimeIndex and the discharge values
    """
    file_path = (Path(__file__).parent / f"../data/hourly_flows_2016-19/flows-{year}.txt").resolve()
    flow_df = pd.read_csv(file_path, delimiter='\t').drop(columns='V')
    flow_df.columns = ['datetime', 'discharge']
    flow_df['datetime'] = pd.to_datetime(flow_df['datetime'], dayfirst=True)
    flow_df = flow_df.set_index('datetime')
    return flow_df


def get_combined_flow():

    """ Combining dataframes from individual years into one dataframe.

       :return: a combined dataframe, with interpolated missing values.
    """

    dfs = []
    for year in range(2016, 2021):
        dfs.append(get_yearly_flow(year=year))
    flow_df = pd.concat(dfs)
    flow_df = flow_df.asfreq('H').interpolate()
    return flow_df


def get_combined_flow_split():

    """
    Splitting the combined data of four years into training, validation and test sets.

    :return: three dataframes
    """

    flow_df = get_combined_flow()
    train = flow_df.loc[flow_df.index < pd.to_datetime('2018-07-01 00:00:00')]
    validation = flow_df.loc[(flow_df.index > pd.to_datetime('2018-07-01 00:00:00')) & (
                flow_df.index < pd.to_datetime('2019-07-01 00:00:00'))]
    test = flow_df.loc[flow_df.index > pd.to_datetime('2019-07-01 00:00:00')]
    return train, validation, test
