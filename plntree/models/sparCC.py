import numpy as np

from plntree.utils.functions import clr_transform, invert_clr_transform, seed_all


class SparCC:
    def __init__(self):
        self.Sigma = None
        self.mu = None

    def fit(self, X):
        X_clr = clr_transform(X)
        self.Sigma = np.cov(X_clr, rowvar=False)
        self.mu = X_clr.mean(0)
        return self

    def sample(self, batch_size, seed=None):
        """Sample from the multivariate normal distribution."""
        seed_all(seed)
        X_clr = np.random.multivariate_normal(self.mu, self.Sigma, size=batch_size)
        return invert_clr_transform(X_clr)
