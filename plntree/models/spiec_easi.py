import numpy as np
from plntree.utils import seed_all
from sklearn.covariance import GraphicalLasso
from plntree.utils.functions import clr_transform, invert_clr_transform


class SPiEC_Easi:
    def __init__(self, glasso_args):
        """Initialize the SPiEC-Easi model."""
        self.glasso_args = glasso_args
        self.Omega = None
        self.Sigma = None
        self.mu = None

    def fit(self, X, seed=None):
        """Fit the SPiEC-Easi model to the input data."""
        seed_all(seed)
        # Apply CLR transformation
        X_clr = clr_transform(X)

        # Fit the Graphical Lasso model
        model = GraphicalLasso(**self.glasso_args)
        model.fit(X_clr)

        self.Omega = model.precision_
        self.Sigma = model.covariance_
        self.mu = X_clr.mean(axis=0)
        return self

    def sample(self, batch_size, seed=None):
        """Sample from the multivariate normal distribution."""
        seed_all(seed)
        Z = np.random.multivariate_normal(self.mu, self.Sigma, size=batch_size)
        return invert_clr_transform(Z)