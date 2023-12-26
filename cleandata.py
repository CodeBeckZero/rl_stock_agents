# CSV - Input and Wrangling

import numpy as np
import pandas as pd

def NASDAQ_csv_input(file_name,file_path):
    """
    ---------------------------------------------------------------------------------------------
    Converts NASDAQ stock csv files from https://www.nasdaq.com/market-activity/quotes/historical
    to pd.dataframe[date, open, high, low, close, volume] 
    with dtypes[Datetime,np.float32, np.float32, np.float32, np.float32, np.float32, np.int)
     in ascentding order
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
    file_name: string name of full file name
    file_path: string name of full path to file
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
    pd.dataframe
    """
    # Import File
    df_ohlcv = pd.read_csv(f'{file_path}/{file_name}').iloc[::-1].reset_index()

    # Updating Column names and order
    column_names_mapping = {'Date':'date',
                            'Close/Last':'close',
                            'Volume':'volume',
                            'Open':'open',
                            'High':'high',
                            'Low':'low'}
    desired_order = ['date','open','high','low','close','volume']
    df_ohlcv = df_ohlcv.rename(columns=column_names_mapping)[desired_order]


    # Converting to Date String to datetime datatype
    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'] , format='%m/%d/%Y')

    # Converting currency columns to float32 datatype
    columns_with_dollars = [col for col in column_names_mapping.values() if col not in ['date','volume']]

    for col in columns_with_dollars:
        df_ohlcv[col] = df_ohlcv[col].str.replace('$', '').astype(float)

    return df_ohlcv
                            