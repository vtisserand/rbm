import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM

rbm = BernoulliRBM(n_components=256, 
                   learning_rate=0.1, 
                   batch_size=10, 
                   n_iter=20, 
                   random_state=42, 
                   verbose=True)
# TODO: add Pipeline with GridSearch for best params etc.
# rbm.fit(features.to_numpy())

def generate_synthetic_samples(rbm: BernoulliRBM, n_samples: int=1, n_gibbs_steps: int=100):
    """
    Generate synthetic samples using Gibbs sampling.

    Parameters:
      rbm: BernoulliRBM
        Trained RBM model.
      n_samples: int
        Number of synthetic samples to generate.
      n_gibbs_steps: int
        Number of Gibbs sampling steps to perform.

    Returns:
      synthetic_samples: ndarray
        Array of shape (n_samples, n_features) containing synthetic samples.
    """
    synthetic_samples = np.random.rand(n_samples, rbm.n_features_in_) < 0.5  # Initialize with random binary values

    for _ in range(n_gibbs_steps):
        synthetic_samples = rbm.gibbs(synthetic_samples)

    return synthetic_samples

def get_synthetic_series(rbm: BernoulliRBM, n_samples: int=10):
    synth = generate_synthetic_samples(rbm, n_samples=n_samples)
    synth = pd.DataFrame(synth)
    synth = synth.astype(int)