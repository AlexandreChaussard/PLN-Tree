from abc import ABC, abstractmethod
import numpy as np

from plntree.metrics import utils


class AlphaDiversity(ABC):

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    @abstractmethod
    def compute_batch(self, batch):
        pass


class LayerAlphaDiversity(AlphaDiversity):

    def __init__(self, taxonomy, layer):
        super(LayerAlphaDiversity, self).__init__(taxonomy)
        self.layer = layer

    @abstractmethod
    def compute(self, counts):
        pass

    def compute_batch(self, batch):
        K_l = self.taxonomy.getLayersWidth()[self.layer]
        X_l = batch[:, self.layer, :K_l]
        values = np.zeros(X_l.size(0))
        for i, X_il in enumerate(X_l):
            values[i] = self.compute(X_il.detach().numpy())
        return values


class Shannon(LayerAlphaDiversity):

    def compute(self, counts, base=2):
        freqs = counts / counts.sum()
        nonzero_freqs = freqs[freqs.nonzero()]
        return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)


class Simpson(LayerAlphaDiversity):

    def compute(self, counts):
        freqs = counts / counts.sum()
        dominance = (freqs * freqs).sum()
        return dominance


class Gini(LayerAlphaDiversity):

    def compute(self, counts):
        freqs = counts / counts.sum()
        dominance = (freqs * freqs).sum()
        gini = 1 - dominance
        return gini


class InverseSimpson(LayerAlphaDiversity):

    def compute(self, counts):
        freqs = counts / counts.sum()
        dominance = (freqs * freqs).sum()
        return 1 / dominance


class Chao1(LayerAlphaDiversity):

    def compute(self, counts, bias_corrected=False):
        o, s, d = utils.OSD(counts)
        o, s, d = float(o), float(s), float(d)
        if not bias_corrected and s and d:
            return o + s ** 2. / (2. * d)
        else:
            return o + s * (s - 1.) / (2. * (d + 1))


class FaithPD(AlphaDiversity):

    def compute(self, dataloader):
        # (branch_lengths * (counts_by_node > 0)).sum()
        ...
