import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from typing import List

def QQ_plot(CCY_pairs, log_return_obs, log_return_sim, log_return_rbm):

    """
    QQ-Plot log return comparisons for a list of currency pairs against observed data, simulated returns,
    and Bernoulli RBM sample.
    
    """
    n_pairs = len(CCY_pairs)
    fig, axes = plt.subplots(nrows=n_pairs, ncols=3, figsize=(18, 6 * n_pairs))  # 3 plots per row
    
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')
        
    for i, pair in enumerate(CCY_pairs):
        pair_log_ret = pair + '_log_ret'
        
        x_data = np.sort(log_return_obs[pair_log_ret])
        y_data = np.sort(log_return_sim[pair_log_ret].sample(len(log_return_obs[pair_log_ret])))
        axes[i, 0].scatter(x_data, y_data, color = 'royalblue')
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        axes[i, 0].plot(x_data, intercept + slope*x_data, 'royalblue', label='Fit line')
        axes[i, 0].set_title(f'{pair_log_ret}: Data vs. Normal')
        axes[i, 0].set_xlabel('Theoretical quantiles')
        axes[i, 0].set_ylabel('Ordered Values')
        axes[i, 0].grid(True)
    
        x_data = np.sort(log_return_rbm[pair_log_ret])
        y_data = np.sort(log_return_sim[pair_log_ret].sample(len(log_return_rbm[pair_log_ret])))
        axes[i, 1].scatter(x_data, y_data, color = 'royalblue')
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        axes[i, 1].plot(x_data, intercept + slope*x_data, 'royalblue', label='Fit line')
        axes[i, 1].set_title(f'{pair_log_ret}: Bernoulli RBM vs. Normal')
        axes[i, 1].set_xlabel('Theoretical quantiles')
        axes[i, 1].set_ylabel('Ordered Values')
        axes[i, 1].grid(True)
    
        x_data = np.sort(log_return_rbm[pair_log_ret])
        y_data = np.sort(log_return_obs[pair_log_ret].sample(len(log_return_rbm[pair_log_ret])))
        axes[i, 2].scatter(x_data, y_data, color = 'royalblue')
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        axes[i, 2].plot(x_data, intercept + slope*x_data, 'royalblue', label='Fit line')
        axes[i, 2].set_title(f'{pair_log_ret}: Bernoulli RBM vs. Data')
        axes[i, 2].set_xlabel('Theoretical quantiles')
        axes[i, 2].set_ylabel('Ordered Values')
        axes[i, 2].grid(True)
    
    
    plt.tight_layout()
    plt.show()

from src.data_utils import tail_dependence_function
def upper_lower_tails(CCY_pairs, log_return_obs, log_return_sim, log_return_rbm):
    
    """
    This function analyzes and visualizes the tail dependence between pairs of currency log returns.
    """
    
    quantiles = np.linspace(0, 0.5, 100)
    combined_quantiles = np.linspace(0, 1, 2 * len(quantiles))
    n_pairs = len(CCY_pairs) 
    n_rows = n_pairs // 2 + (n_pairs % 2 > 0)
    plt.figure(figsize=(2 * 5, n_rows * 4))
    
    
    for i, (CCY1, CCY2) in enumerate(CCY_pairs, 1):
        plt.subplot(n_rows, 2, i)
        
        real_x = log_return_obs[f'{CCY1}_log_ret']
        real_y = log_return_obs[f'{CCY2}_log_ret']
        gen_x = log_return_rbm[f'{CCY1}_log_ret']
        gen_y = log_return_rbm[f'{CCY2}_log_ret']
        param_x = log_return_sim[f'{CCY1}_log_ret']
        param_y = log_return_sim[f'{CCY2}_log_ret']
    
        lower_tail_real, upper_tail_real = tail_dependence_function(real_x, real_y, quantiles)
        combined_tails_real = lower_tail_real + upper_tail_real
        lower_tail_gen, upper_tail_gen = tail_dependence_function(gen_x, gen_y, quantiles)
        combined_tails_gen = lower_tail_gen + upper_tail_gen
        
        lower_tail_sim, upper_tail_sim = tail_dependence_function(param_x, param_y, quantiles)
        combined_tails_sim = lower_tail_sim + upper_tail_sim
        
        plt.plot(combined_quantiles, combined_tails_real, label='Real Data')
        plt.plot(combined_quantiles, combined_tails_gen, label='Synthetic', linestyle='--')
        plt.plot(combined_quantiles, combined_tails_sim, label='Parametric')
        plt.title(f' {CCY1}/{CCY2}')
        plt.xlabel('Quantiles')
        plt.ylabel('Tail Dependence Probability')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()