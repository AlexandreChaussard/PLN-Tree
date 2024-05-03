import torch
import torch.nn as nn
import numpy as np

from enum import Enum

from plntree.utils.model_utils import Embedder, BoundLayer, Preprocessing, progressive_NN


class VariationalApproximation(Enum):
    MEAN_FIELD = "mean_field"
    BACKWARD = "backward"
    BRANCH = "branch"


def mean_field(K, effective_K, n_variational_layers, proportion=False):
    # TODO: log transform in the non-offset case seems to decrease quality of the results, but why?
    m_fun = nn.ModuleList([
        nn.Sequential(
            Preprocessing(K[layer], log_transform=False, proportion=proportion, standardize=proportion),
            *progressive_NN(K[layer], effective_K[layer], n_variational_layers),
            BoundLayer(-100, 25, smoothing_factor=0.05)
        )
        for layer, _ in enumerate(effective_K)
    ])
    S_fun = nn.ModuleList([
        nn.Sequential(
            Preprocessing(K[layer], log_transform=False, proportion=proportion, standardize=proportion),
            *progressive_NN(K[layer], effective_K[layer], n_variational_layers),
            BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
        )
        for layer, _ in enumerate(effective_K)
    ])
    return m_fun, S_fun


def backward_markov(input_size, effective_K, embedder_type, embedding_size, n_embedding_layers=2, n_embedding_neurons=32, embedder_dropout=0.1, n_after_layers=1):
    # The first layer is only getting X^{1:L}
    # The other layers are getting X^{1:l} and Z^{l + 1}
    # In the end, the size of the input "l" is K_{l + 1} + sum_{j=1}^{l} K_{j}

    # To stabilize the learning, we embed the X^{1:l} in the same space as E^{l} = f(X^{1:l})
    # this is performed by a RNN layer
    # Consequently, each layer is getting the following input size: embedding_size + K_{l + 1}
    embedder = Embedder(
        input_size=input_size,
        embedding_size=embedding_size,
        hidden_size=n_embedding_neurons,
        n_layers=n_embedding_layers,
        recurrent_network=embedder_type,
        dropout=embedder_dropout
    )
    m_fun_list = []
    S_fun_list = []
    for i in range(len(effective_K)):
        # The input size of X^{1:l} would be sum(self.K[:i+1])
        # But it's simpler now with the embedding!
        input_size = embedding_size
        # If we are not at the last layer, the model also gets Z^{l+1} as input
        if i != len(effective_K) - 1:
            input_size += effective_K[i + 1]
        m_fun_list.append(
            nn.Sequential(
                *progressive_NN(input_size, effective_K[i], n_after_layers),
                BoundLayer(-100, 25, smoothing_factor=0.05)
            )
        )
        S_fun_list.append(
            nn.Sequential(
                *progressive_NN(input_size, effective_K[i], n_after_layers),
                BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
            )
        )

    m_fun = nn.ModuleList(m_fun_list)
    S_fun = nn.ModuleList(S_fun_list)
    return m_fun, S_fun, embedder


class BranchMarkovLayer(nn.Module):

    def __init__(self, taxonomy, layer, selected_layers, mask, mean=True, n_layers=2):
        super().__init__()
        self.taxonomy = taxonomy
        self.layer = layer
        self.selected_layers = selected_layers
        self.return_mean = mean
        self.mask = mask

        self.m = []
        self.log_S = []
        for node in taxonomy.getNodesAtDepth(layer + selected_layers[0]):
            if mask[node.layer_index] is False:
                continue
            children = node.children
            # The input are C(Z_k^{l}) and the branch from X_k^{l} to the root
            # So the input size is len(children) + depth + 1 (since depth starts at 0)
            input_size = len(children) + node.depth + 1 - self.selected_layers[0]
            # The output is just a value (either m or log_S)
            self.m.append(nn.Sequential(
                *progressive_NN(input_size, 1, n_layers),
                BoundLayer(-12., 12., smoothing_factor=0.2)
            ))
            self.log_S.append(nn.Sequential(
                *progressive_NN(input_size, 1, n_layers),
                BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
            ))
        self.m = nn.ModuleList(self.m)
        self.log_S = nn.ModuleList(self.log_S)

    def forward(self, X_1tol, Z_l_next):
        m = []
        log_S = []
        i = 0
        for node in self.taxonomy.getNodesAtDepth(self.layer + self.selected_layers[0]):
            if self.mask[node.layer_index] is False:
                continue
            children = node.children
            # Fetch the latent children of the node
            children_index = [child.layer_index for child in children]
            Z_children = Z_l_next[:, children_index]
            # Fetch the abundance branch of the node
            X_branch = [X_1tol[:, self.layer, node.layer_index].unsqueeze(-1)]
            parent = node.parent
            while parent is not None and parent.depth >= min(self.selected_layers):
                parent_layer = parent.depth - self.selected_layers[0]
                X_branch.append(X_1tol[:, parent_layer, parent.layer_index].unsqueeze(-1))
                parent = parent.parent
            X_branch = torch.cat(X_branch, dim=1)
            preprocessing = Preprocessing(X_branch.size(1), log_transform=True, standardize=True)
            X_branch = preprocessing(X_branch)
            data = torch.cat([Z_children, X_branch], dim=1)
            m += [self.m[i](data)]
            log_S += [self.log_S[i](data)]
            i += 1

        # Either we return the mean or the log_S
        if self.return_mean:
            return torch.cat(m, dim=1)
        return torch.cat(log_S, dim=1)


def backward_branch_markov(layer_masks, taxonomy, selected_layers):
    # The last layer predicts Z^L from X^{1:L}, but we'll just give it X^L for now
    # Then for each node we have a NN predicting Z_k^l from C(Z_k^{l}) and the branch from X_k^{l} to the root

    m_fun_list = []
    S_fun_list = []
    effective_K = [sum(mask) for mask in layer_masks]
    K = [len(mask) for mask in layer_masks]
    for layer in range(len(effective_K)):
        # Last layer
        if layer == len(effective_K) - 1:
            m_fun_list.append(
                nn.Sequential(
                    nn.Linear(K[layer], effective_K[layer]),
                    BoundLayer(-12., 12., smoothing_factor=0.2)
                )
            )
            S_fun_list.append(
                nn.Sequential(
                    nn.Linear(K[layer], effective_K[layer]),
                    BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
                )
            )
        else:
            m_fun_list.append(BranchMarkovLayer(taxonomy, layer, selected_layers, layer_masks[layer], mean=True))
            S_fun_list.append(BranchMarkovLayer(taxonomy, layer, selected_layers, layer_masks[layer], mean=False))

    m_fun = nn.ModuleList(m_fun_list)
    S_fun = nn.ModuleList(S_fun_list)
    return m_fun, S_fun
