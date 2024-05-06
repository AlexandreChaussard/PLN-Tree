import numpy as np
import torch
import networkx as nx

import plntree.metrics.utils as utils


class MeanAbsoluteError(utils.GraphDistanceMetric):
    def compute(self, taxa_1, taxa_2):
        return torch.mean(torch.mean(torch.abs(taxa_1 - taxa_2), dim=0))


class SpectralDistance(utils.GraphDistanceMetric):

    def __init__(self, taxonomy, method="adjacency", norm=2, binary=False):
        super().__init__(taxonomy)
        self.method = method
        self.norm = norm
        self.binary = binary

    def compute(self, taxa_1, taxa_2):
        A_1 = torch.tensor(utils.get_measured_matrix(taxa_1, self.taxonomy, method=self.method, binary=self.binary))
        A_2 = torch.tensor(utils.get_measured_matrix(taxa_2, self.taxonomy, method=self.method, binary=self.binary))
        spec_1 = torch.linalg.eigvals(A_1)
        spec_2 = torch.linalg.eigvals(A_2)
        value = torch.linalg.norm(spec_1 - spec_2, ord=self.norm)
        return value


class CosineSimilarity(utils.GraphDistanceMetric):

    def __init__(self, taxonomy, method="adjacency", binary=False):
        super().__init__(taxonomy)
        self.method = method
        self.binary = binary

    def compute(self, taxa_1, taxa_2):
        A_1 = utils.get_measured_matrix(taxa_1, self.taxonomy, method=self.method, binary=self.binary)
        A_2 = utils.get_measured_matrix(taxa_2, self.taxonomy, method=self.method, binary=self.binary)
        norm_1 = np.linalg.norm(A_1, ord='fro')
        norm_2 = np.linalg.norm(A_2, ord='fro')
        if norm_1 == norm_2 and norm_1 == 0:
            return 1
        elif norm_1 == 0 or norm_2 == 0:
            return 0
        value = np.sqrt(np.trace(A_1.T @ A_2)) / (np.linalg.norm(A_1, ord='fro') * np.linalg.norm(A_2, ord='fro'))
        return value


class GraphEditDistance(utils.GraphDistanceMetric):

    def compute(self, taxa_1, taxa_2):
        A_1 = utils.get_measured_matrix(taxa_1.numpy(), self.taxonomy, method="adjacency", binary=True)
        A_2 = utils.get_measured_matrix(taxa_2.numpy(), self.taxonomy, method="adjacency", binary=True)
        G_1 = nx.from_numpy_array(A_1)
        G_2 = nx.from_numpy_array(A_2)
        return nx.graph_edit_distance(G_1, G_2)
