import numpy as np
import pandas as pd
from scipy.stats import norm

def log_returns(data):
    return np.log(data / data.shift(1))

def drift_calc(data):
    lr = log_returns(data)
    drift = lr.mean() - 0.5 * lr.var()
    return drift.values

def simulate_correlated_returns(data, days=100, iterations=1000):
    """ 
    Simulate correlated return paths using Cholesky decomposition and Monte Carlo simulation.

    """
    log_ret = log_returns(data)
    drift = drift_calc(data)
    stdev = log_ret.std()

    # Cholesky decomposition of the covariance matrix
    chol = np.linalg.cholesky(log_ret.cov())

    simulated_paths = np.zeros((days + 1, len(data.columns), iterations))
    simulated_paths[0] = data.iloc[-1].to_numpy().reshape(-1, 1)  

    for i in range(iterations):
        Z = norm.rvs(size=(days, len(data.columns)))
        daily_returns = chol @ Z.T  # Cholesky matrix to introduce correlation
        daily_returns = np.exp(drift.reshape(-1, 1) + stdev.values.reshape(-1, 1) * daily_returns)
        for t in range(1, days + 1):
            simulated_paths[t, :, i] = simulated_paths[t-1, :, i] * daily_returns[:, t-1]

    simulation_results = {}
    for idx, ticker in enumerate(data.columns):
        simulation_results[ticker] = simulated_paths[:, idx, :].flatten() 

    return pd.DataFrame(simulation_results)