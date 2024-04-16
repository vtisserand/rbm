import numpy as np
import pandas as pd
import yfinance as yf

def get_fx_data(pairs: list[str], start_date: str='2020-01-01', end_date: str='2023-12-31', df: bool=False) -> dict:
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
    for instr in df.columns:
        df["{}_log_ret".format(instr)] = np.log(df["{}".format(instr)] / df["{}".format(instr)].shift(1))
    return df.iloc[1:]

    
def to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    To map log returns to a 16-bits digit, we compute the spread between the current value
    and the minimum value amongst all the samples, and get a [0,1]-valued float computing
    (X_curr - X_min) / (X_max - X_min). Then, multiply by 65535 (2^16 - 1).
    """
    EPSILON = np.finfo(float).eps

    ccy_pairs = set([s.split('_')[0] for s in df.columns])
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

    return pd.concat(dfs, axis=1).dropna()