# Synthetic FX data with generative models

## Restricted Boltzmann machine

![A sampled market](https://github.com/vtisserand/rbm/blob/main/report/img/output.gif)

Example use:
```python
from sklearn.neural_network import BernoulliRBM

from src.data_utils import get_fx_data, compute_log_ret, to_binary, to_float, get_min_max_info
from src.models.rbm import get_synthetic_series

# Gerring and transforming the data
df = get_fx_data(pairs=["EURUSD", "GBPUSD", "EURGBP"], start_date='2020-01-01', end_date='2023-12-31')

log_ret_df = compute_log_ret(df)
features = to_binary(log_ret_df)

# The actual model
rbm = BernoulliRBM(n_components=256, 
                   learning_rate=0.01, 
                   batch_size=50, 
                   n_iter=50, 
                   random_state=42, 
                   verbose=1)

rbm.fit(features.to_numpy())

# Genering data
get_synthetic_series(rbm=rbm, log_ret_df=log_ret_df, pairs=features.columns, n_samples=300)
```