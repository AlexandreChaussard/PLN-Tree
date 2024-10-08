from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from skbio.stats.distance import permanova, permdisp
from skbio.stats.distance import DistanceMatrix

from plntree.utils import tree_utils, seed_all


def observed_otus(counts):
    """
    Calculate the number of distinct OTUs.
    """
    return (counts != 0).sum()


def singles(counts):
    """
    Calculate number of single occurrences (singletons).
    """
    return (counts == 1).sum()


def doubles(counts):
    """
    Calculate number of double occurrences (doubletons).
    """
    return (counts == 2).sum()


def OSD(counts):
    """
    Calculate observed OTUs, singles, and doubles.
    """
    return observed_otus(counts), singles(counts), doubles(counts)


def get_measured_matrix(taxa, taxonomy, method="adjacency", binary=False):
    A = tree_utils.abundance_tree_builder(taxonomy, taxa).to_adjacency_matrix(binary)
    if method == "adjacency":
        G = A
    elif method == "laplacian":
        G = tree_utils.abundance_tree_builder(taxonomy, taxa).effective_degree_matrix() - A
    elif method == "unsigned_laplacian":
        G = tree_utils.abundance_tree_builder(taxonomy, taxa).effective_degree_matrix() + A
    return G

def bootstrap_mean_project_beta_diversity(beta, beta_args, X_base, X_list, names, n_samples=100, n_repeat=10, seed=None):
    seed_all(seed)
    values = np.zeros((n_repeat, len(X_list)+1))
    indices = np.arange(0, len(X_base))
    X_base_norm = X_base / X_base.sum(-1, keepdims=True)
    X_list_norm = [X_ / X_.sum(-1, keepdims=True) for X_ in X_list]
    for i in range(n_repeat):
        np.random.shuffle(indices)
        intra = beta.compute_batch(X_base_norm[indices[:n_samples]], X_base_norm[indices[:n_samples]], **beta_args).min(1).mean()
        values[i][0] = intra
        for j in range(1, len(X_list)+1):
            values[i][j] = beta.compute_batch(X_base_norm[indices[:n_samples]], X_list_norm[j-1][indices[:n_samples]], **beta_args).min(1).mean()
    df = pd.DataFrame()
    for i, name in enumerate(names):
        df[name] = [values[:, i].mean(), values[:, i].std()]
    df.index = ['mean', 'std']
    return df.T

def bootstrap_relative_mean_coverage_beta_diversity(beta, beta_args, X_base, X_list, names, n_samples=100, n_repeat=10, seed=None):
    seed_all(seed)
    values = np.zeros((n_repeat, len(X_list)+1))
    indices = np.arange(0, len(X_base))
    X_base_norm = X_base / X_base.sum(-1, keepdims=True)
    X_list_norm = [X_ / X_.sum(-1, keepdims=True) for X_ in X_list]
    for i in range(n_repeat):
        np.random.shuffle(indices)
        intra = beta.compute_batch(X_base_norm[indices[:n_samples]], X_base_norm[indices[:n_samples]], **beta_args).mean(1).mean()
        values[i][0] = intra
        for j in range(1, len(X_list)+1):
            values[i][j] = beta.compute_batch(X_list_norm[j-1][indices[:n_samples]], X_list_norm[j-1][indices[:n_samples]], **beta_args).mean(1).mean()
        values[i, :] = np.abs(values[i, :] - intra) / intra
    df = pd.DataFrame()
    for i, name in enumerate(names):
        df[name] = [values[:, i].mean(), values[:, i].std()]
    df.index = ['mean', 'std']
    return df.T

def bootstrap_perm_pvalue(method, beta, X_base, X_list, names, n_samples=100, n_repeat=10, permutations=1000, beta_args={}, seed=None, replacement=False):
    seed_all(seed)
    values = {name:[0]*n_repeat for name in names}
    indices = np.arange(0, len(X_base))
    np.random.shuffle(indices)
    X_base_norm = X_base / X_base.sum(-1, keepdims=True)
    X_list_norm = [X_ / X_.sum(-1, keepdims=True) for X_ in X_list]
    if not replacement and n_samples * n_repeat > len(X_base):
        raise ValueError('n_samples * n_repeat should be less than the number of samples in the compared datasets')
    for i in range(n_repeat):
        if replacement:
            indices_selected = np.random.choice(indices, n_samples, replace=False)
        else:
            indices_selected = indices[n_samples*i:n_samples*(i+1)]
        for j in range(len(X_list)):
            dist_matrix = beta.dissimilarity((X_base_norm[indices_selected], X_list_norm[j][indices_selected]), **beta_args)
            dist_matrix = DistanceMatrix(dist_matrix)
            grouping = [0] * n_samples + [1] * n_samples
            if method == 'permanova':
                values[names[j]][i] = permanova(dist_matrix, grouping, permutations=permutations)['p-value']
            elif method == 'permdisp':
                values[names[j]][i] = permdisp(dist_matrix, grouping, permutations=permutations)['p-value']
    return pd.DataFrame(values)

def threshold_bootstrap_pvalues(pvalues, alpha=0.95):
    thresh = pvalues.apply(lambda p: p > alpha)
    thresh_str = [f"{np.round(thresh.mean(0)[i], 3)} ({np.round(thresh.std(0)[i],3)})" for i in range(len(thresh.mean(0)))]
    df = pd.DataFrame(data=thresh_str, index=thresh.columns)
    return df

def fisher_method_bootstrap_values(pvalues):
    df = pvalues.copy()
    df = -2*df.apply(lambda p: np.log(p + 1e-30))
    mean_fisher = df.mean(0)
    std_fisher = df.std(0)
    res = [f'{np.round(mean_fisher[i], 3)} ({np.round(std_fisher[i],3)})' for i in range(len(mean_fisher))]
    res_df = pd.DataFrame()
    res_df['mean fisher'] = res
    res_df.index = df.columns
    return res_df

class GraphDistanceMetric(ABC):
    def __init__(self, taxonomy, layer_shift=0):
        self.taxonomy = taxonomy
        self.layer_shift = layer_shift

    @abstractmethod
    def compute(self, taxa_1, taxa_2):
        pass

    def compute_batch(self, batch_1, batch_2):
        values = np.zeros((len(batch_1), len(batch_2)))
        for i, X_i in enumerate(batch_1):
            for j, X_j in enumerate(batch_2):
                values[i][j] = self.compute(X_i, X_j)
        return values

    def compute_self_batch(self, batch):
        return self.compute_batch(batch, batch)