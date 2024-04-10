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

    
def to_binary(df: pd.DataFrame):
    ccy_pairs = set([s.split('_')[0] for s in df.columns])
    for instr in ccy_pairs:
        df["{}_log_ret_binary".format(instr)] = df["{}_log_ret".format(instr)].apply(lambda x: format(int(x * 1e15), '016b'))
    return df