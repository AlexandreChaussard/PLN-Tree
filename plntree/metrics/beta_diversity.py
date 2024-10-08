from abc import ABC, abstractmethod

import numpy as np
import torch

from plntree.metrics import utils


class LayerBetaDiversity(ABC):

    def __init__(self, K):
        self.K = K

    @abstractmethod
    def compute(self, counts_1, counts_2):
        pass

    def compute_batch(self, batch_1, batch_2, layer):
        K_l = self.K[layer]
        X_l_1 = batch_1[:, layer, :K_l]
        X_l_2 = batch_2[:, layer, :K_l]
        distances = np.zeros((len(X_l_1), len(X_l_2)))
        for i, X_il_1 in enumerate(X_l_1):
            for j, X_jl_2 in enumerate(X_l_2):
                distances[i][j] = self.compute(X_il_1, X_jl_2)
        return distances

    def dissimilarity(self, list_batch, layer):
        X = np.concatenate(list_batch, axis=0)
        distances = np.identity(len(X))
        for i in range(len(X)):
            for j in range(i, len(X)):
                distances[i, j] = self.compute(X[i, layer, :self.K[layer]], X[j, layer, :self.K[layer]])
        distances = distances + np.triu(distances, 1).T
        return distances


class BrayCurtis(LayerBetaDiversity):

    def compute(self, counts_1, counts_2):
        assert (len(counts_1) == len(counts_2))
        observed_OTUs_1 = np.array(counts_1[(counts_1 > 0) & (counts_2 > 0)])
        observed_OTUs_2 = np.array(counts_2[(counts_1 > 0) & (counts_2 > 0)])
        li = [np.min((observed_OTUs_1[i], observed_OTUs_2[i])) for i in range(len(observed_OTUs_1))]
        C = np.sum(li)
        S_1 = np.sum(observed_OTUs_1)
        S_2 = np.sum(observed_OTUs_2)
        if S_1 + S_2 == 0:
            return 1
        return 1 - 2 * C / (S_1 + S_2)


class Jaccard(utils.GraphDistanceMetric):

    def __init__(self, taxonomy, layer_shift=0, weighted=False):
        super().__init__(taxonomy)
        self.weighted = weighted
        self.layer_shift = layer_shift

    def compute(self, taxa_1, taxa_2):
        n_shared = 0
        n_tot = 0
        for layer in range(0, len(taxa_1)):
            K_l = self.taxonomy.getLayersWidth()[layer + self.layer_shift]
            X_l_1 = taxa_1[layer, :K_l]
            X_l_2 = taxa_2[layer, :K_l]
            otu_l1 = X_l_1[(X_l_1 > 0) & (X_l_2 > 0)]
            otu_l2 = X_l_2[(X_l_1 > 0) & (X_l_2 > 0)]
            if self.weighted:
                n_shared += np.sum([np.min((otu_l1[i], otu_l2[i])) for i in range(len(otu_l1))])
                n_tot += np.sum([np.max((otu_l1[i], otu_l2[i])) for i in range(len(otu_l1))])
            else:
                # Cardinal of the intersection
                n_shared += len(otu_l1)
                # Cardinal of the union
                X_l_union = X_l_1 + X_l_2
                n_tot += len(X_l_union[X_l_union > 0])
        # If the union of both is empty, it means both samples are empty and therefore equal
        # Hence to prevent division by zero, we add that specific test
        if n_tot == n_shared:
            return 0
        return 1 - n_shared / n_tot


class UniFrac(utils.GraphDistanceMetric):

    def __init__(self, taxonomy, layer_shift=0, weighted=False):
        super().__init__(taxonomy)
        self.weighted = weighted
        self.layer_shift = layer_shift

    def compute(self, taxa_1, taxa_2):
        unifrac = 0
        branch_total = 0
        X_1 = torch.tensor(taxa_1)
        X_2 = torch.tensor(taxa_2)
        for node in self.taxonomy.nodes:
            layer = node.depth - self.layer_shift
            a = X_1[layer, node.layer_index]
            b = X_2[layer, node.layer_index]
            if self.weighted:
                if layer > 0:
                    a /= X_1[layer - 1, node.parent.layer_index]
                    b /= X_2[layer - 1, node.parent.layer_index]
                # The weight of the branch of a node-node.parent corresponds to the value of the node
                branch_length = node.value
            else:
                a = (a > 0) * 1
                b = (b > 0) * 1
                # Unweighted branches
                branch_length = 1
            unifrac += branch_length * np.abs(a - b)
            branch_total += branch_length * np.max((a, b))
        if self.weighted:
            # The total weight is the sum of weights from the leaves to the root
            branch_total = 0
            for leaf in self.taxonomy.getNodesAtDepth(len(taxa_1)-1):
                branch_total += leaf.value
                node = leaf.parent
                while node is not None:
                    branch_total += node.value
                    node = node.parent
        if unifrac == branch_total:
            return 1
        return unifrac / branch_total
