import numpy as np
import pandas as pd
import yfinance as yf

def get_fx_data(pairs: list[str], start_date: str='2020-01-01', end_date: str='2023-12-31', df: bool=True) -> dict:
    ccy_pairs = [pair + '=X' for pair in pairs]

    start_date = start_date
    end_date = end_date

    closing_prices = {}
    dfs = []
    for pair in ccy_pairs:
        data = yf.download(pair, start=start_date, end=end_date)
        closing_prices[pair] = data.Close.values.tolist()
        dfs.append(data[["Close"]])

    # Careful with FX conventions around the world
    if df:
        merged_data = pd.concat(dfs, axis=1).dropna()
        return merged_data.set_axis(pairs, axis=1)
    
    return closing_prices

def compute_log_ret(df: pd.DataFrame):
    result_df = df.copy()
    for instr in df.columns:
        result_df["{}_log_ret".format(instr)] = np.log(result_df["{}".format(instr)] / result_df["{}".format(instr)].shift(1))
    return result_df.iloc[1:]

def _get_ccy_pairs(df: pd.DataFrame) -> list:
    """
    From a dataframe with column names, retrieve an ordered set or the currency pairs monitored.
    """
    return sorted(set([s.split('_')[0] for s in df.columns]), key=df.columns.tolist().index)

def to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    To map log returns to a 16-bits digit, we compute the spread between the current value
    and the minimum value amongst all the samples, and get a [0,1]-valued float computing
    (X_curr - X_min) / (X_max - X_min). Then, multiply by 65535 (2^16 - 1).
    """
    EPSILON = np.finfo(float).eps

    ccy_pairs = sorted(set([s.split('_')[0] for s in df.columns]), key=df.columns.tolist().index) # To keep the order
    dfs = []
    for instr in ccy_pairs:
        curr_samples = df["{}_log_ret".format(instr)]
        
        X_min = curr_samples.min() - EPSILON
        X_max = curr_samples.max() + EPSILON
        
        X_integer = ((curr_samples - X_min) / (X_max - X_min) * 65535).astype(int)
        X_binary = X_integer.apply(lambda x: format(x, '016b'))
        
        binary_split = X_binary.apply(list)
        
        binary_df = pd.DataFrame(binary_split.tolist(), columns=[f'{instr}_{i:02}' for i in range(1, 17)])
        
        dfs.append(binary_df)

    return pd.concat(dfs, axis=1).dropna().astype(int)

def get_min_max_info(df: pd.DataFrame) -> dict:
    """
    We need to store the min-max log returns for each instrument.
    """
    EPSILON = np.finfo(float).eps
    ccy_pairs = sorted(set([s.split('_')[0] for s in df.columns]), key=df.columns.tolist().index)
    min_max = dict()
    for instr in ccy_pairs:
        curr_samples = df["{}_log_ret".format(instr)]
        X_min = curr_samples.min() - EPSILON
        X_max = curr_samples.max() + EPSILON
        min_max[instr] = [X_min, X_max]
    return min_max

def to_float(df: pd.DataFrame, min_max: dict) -> pd.DataFrame:
    """
    Reverse process: from 16-bits binary data, get the log returns.
    We need additional info: to map everything back to consistent returns,
    pass the min-max log returns per instrument (e.g. USDBRL moves more than USDJPY).
    """
    # Group the columns e.g. EURUSD_01, ..., EURUSD_16 belong to the EURUSD variable.
    ccy_pairs = list(sorted(set([s.split('_')[0] for s in df.columns]), key=[x[0] for x in df.columns.str.split('_')].index))

    grouped = df.groupby(lambda x: x.split('_')[0], axis=1)
    for ccy, group in grouped:
        df[f'{ccy}'] = group.apply(lambda row: ''.join(map(str, row)), axis=1)
    df = df[ccy_pairs]

    df_int = df.applymap(lambda x: int(x, 2))

    dfs = []
    for instr in ccy_pairs:
        curr_samples = df_int[instr]

        X_real = (min_max[instr][0] + curr_samples * (min_max[instr][1] - min_max[instr][0]) / 65535)
        dfs.append(pd.DataFrame(X_real, columns=[instr]))
    
    return pd.concat(dfs, axis=1)

def get_latest_value(df: pd.DataFrame) -> dict:
    """
    Once we have generated log returns, we need to convert the data back to an exchange rate.
    We need the latest traded value to do so.
    """
    ccy_pairs = _get_ccy_pairs(df)
    return df[ccy_pairs].iloc[-1].to_dict()