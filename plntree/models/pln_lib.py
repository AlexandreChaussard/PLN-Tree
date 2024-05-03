import torch
from pyPLNmodels import Pln
import numpy as np


def fit(X, layer, K, tol=1e-15):
    X_l = X[:, layer, :K[layer]].numpy().astype(np.float64)
    pln = Pln(X_l, add_const=True)
    pln.fit(tol=tol, nb_max_iteration=800_000)
    return pln


def mu(pln):
    return pln.coef.reshape(-1, ) + pln.offsets.mean(axis=0)


def sigma(pln):
    return pln.covariance


def omega(pln):
    return torch.inverse(sigma(pln))


def sample(pln, n_samples, offsets=None):
    means = mu(pln)
    cov = sigma(pln)
    X = np.zeros((n_samples, cov.shape[1]), dtype=np.float64)
    Z = np.zeros((n_samples, cov.shape[1]), dtype=np.float64)
    if offsets is None:
        O = np.zeros(n_samples) + pln.offsets.mean(axis=0).mean().numpy()
    else:
        O = offsets
    means -= pln.offsets.mean(axis=0)
    for i in range(n_samples):
        Z_i = np.random.multivariate_normal(
            mean=means + O[i],
            cov=cov
        )
        # Sample X ~ Poisson(Z)
        try:
            X_i = np.random.poisson(np.exp(Z_i.astype(np.float64)))
        except ValueError:
            print("Can not compute Poisson with parameters", Z_i[np.where(Z_i > 43)])
            print("Values will be shifted to 43 to avoid overflow (only the offset is affected).")
            Z_i = Z_i - Z_i.max() + 43
            X_i = np.random.poisson(np.exp(Z_i.astype(np.float64)))

        X[i] = X_i
        Z[i] = Z_i
    return X, Z


def encode(pln):
    M = pln.latent_mean
    S = pln.latent_sqrt_var ** 2
    Z = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[0]):
        Z[i] = np.random.multivariate_normal(
            mean=M[i] + pln.offsets[i],
            cov=np.diag(S[i])
        )
    return Z


def decode(Z):
    X = np.zeros_like(Z, dtype=np.float64)
    for i in range(len(Z)):
        X[i] = np.random.poisson(np.exp(Z[i].astype(np.float64)))
    return X
