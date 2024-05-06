from abc import ABC, abstractmethod

import numpy as np

from plntree.utils import tree_utils


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


class GraphDistanceMetric(ABC):
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

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