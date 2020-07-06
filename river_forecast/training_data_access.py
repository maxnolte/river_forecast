import pandas as pd


def get_yearly_flow(year=2016):
    """ Reading text files with yearly data.
    """
    flow_df = pd.read_csv(f'data/hourly_flows_2016-19/flows-{year}.txt', delimiter='\t').drop(columns='V')
    flow_df.columns = ['datetime', 'discharge']
    flow_df['datetime'] = pd.to_datetime(flow_df['datetime'], dayfirst=True)
    flow_df = flow_df.set_index('datetime')
    return flow_df

def get_combined_flow():
    """ Combining dataframes from individual years into one dataframe.
    """
    dfs = []
    for year in range(2016, 2020):
        dfs.append(get_yearly_flow(year=year))
    flow_df = pd.concat(dfs)
    flow_df = flow_df.asfreq('H').interpolate()
    return flow_df

