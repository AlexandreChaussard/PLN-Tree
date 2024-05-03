import networkx
import numpy as np
import random
import networkx as nx
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_hierarchical_tree(nodes_per_layer,
                               random_strategy=np.random.random,
                               min_children=1,
                               seed=None):
    if seed is not None:
        np.random.seed(seed)
    K = np.array(nodes_per_layer).sum()
    G = np.zeros((K, K))

    for l in reversed(range(1, len(nodes_per_layer))):
        K_l = nodes_per_layer[l]
        K_l_parent = nodes_per_layer[l - 1]
        K_cum_parent = np.array(nodes_per_layer)[:l - 1].sum()
        K_cum = K_cum_parent + K_l_parent
        parents_index = np.arange(K_cum_parent, K_cum)
        children_index = np.arange(K_cum, K_cum + K_l)

        proportion = np.abs(random_strategy(K_l_parent))
        proportion = proportion / proportion.sum()
        counts = proportion * K_l
        counts = np.floor(counts) + min_children
        while counts.sum() < K_l:
            k = np.random.randint(0, len(counts))
            counts[k] += 1
        while counts.sum() > K_l:
            k = np.random.choice(np.where(counts > min_children)[0], 1)[0]
            counts[k] -= 1
        n_children = {parent_index: counts[j] for j, parent_index in enumerate(parents_index)}

        i = 0
        current_parent = parents_index[0]
        for c, child_index in enumerate(children_index):
            if n_children[current_parent] == 0:
                i += 1
                current_parent = parents_index[i]
            G[current_parent][child_index] = 1
            n_children[current_parent] -= 1

    return G


def generate_adjacency_matrix(n_nodes, args=(.5,), method="erdos_renyi", seed=None, returnNetworkx=False):
    """
    Generates a random adjacency matrix
    Parameters
    ----------
    n_nodes: number of nodes
    args: arguments of the method. Refers to https://networkx.org/documentation/stable/reference/generators.html
    method: valued in ["erdos_renyi", "preferential_attachment"]
    seed: random seed
    returnNetworkx

    Returns
    -------

    """
    G = None
    if method == "erdos_renyi":
        G = nx.gnp_random_graph(n_nodes, *args, seed=seed)
    elif method == "preferential_attachment":
        G = nx.barabasi_albert_graph(n_nodes, *args)
    if returnNetworkx:
        return G
    return nx.to_numpy_array(G)


def generate_community_adjacency_matrix(n_nodes_per_community, n_random_edges, method="erdos_renyi", method_args=(.5,),
                                        seed=None):
    if seed is not None:
        np.random.seed(seed)

    G = None
    for n_nodes in n_nodes_per_community:
        C = generate_adjacency_matrix(n_nodes, args=method_args, method=method, returnNetworkx=True)
        if G is None:
            G = C
        else:
            G = networkx.disjoint_union(G, C)

    for _ in range(n_random_edges):
        nonedges = list(nx.non_edges(G))
        if len(nonedges) == 0:
            break
        chosen_nonedge = random.choice(nonedges)
        G.add_edge(chosen_nonedge[0], chosen_nonedge[1])

    return nx.to_numpy_array(G)


def generate_precision_matrix(adjacency_matrix, conditioning=0.1, correlation=0.3):
    omega = correlation * adjacency_matrix
    eigen_values, eigen_vectors = np.linalg.eig(omega)
    return omega + np.identity(len(omega)) * (np.abs(np.min(eigen_values)) + conditioning)


def load_artificial_data(model, n_data, batch_size, seed=None, model_args=None):
    if seed is not None:
        torch.manual_seed(seed)
    if model_args is not None:
        X, Z, O = model.sample(n_data, **model_args)
    else:
        X, Z, O = model.sample(batch_size=n_data)
    dataset = TensorDataset(X, Z, O)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def generate_markov_dirichlet_hierarchical_data(n_samples, tree, selected_layers, Omega, mu, offset_total_count,
                                                offset_probs,
                                                alpha_structures):
    # Computing the counts parameters
    log_a = torch.distributions.MultivariateNormal(torch.tensor(mu), precision_matrix=torch.tensor(Omega)).sample(
        (n_samples,))
    a = torch.exp(log_a)
    pi = torch.nn.functional.softmax(a - torch.max(a), dim=1)
    N = torch.distributions.NegativeBinomial(offset_total_count, offset_probs).sample((n_samples,))
    # Getting the structure of the data
    K = list(tree.getLayersWidth().values())[selected_layers[0]:]
    L = len(K)
    X = torch.zeros((n_samples, L, max(K)))
    # Building the Dirichlet parameters for propagation
    alpha_modules = []
    for l in range(0, L - 1):
        if alpha_structures is not None:
            structure = [K[l]] + alpha_structures[l] + [K[l + 1]]
        else:
            structure = [K[l], K[l + 1]]
        module = torch.nn.Sequential()
        for i in range(len(structure) - 1):
            module.append(torch.nn.Linear(structure[i], structure[i + 1]))
            module.append(torch.nn.Softplus())
        alpha_modules.append(module)
    # Draw the first layer
    for i, n in enumerate(N):
        X[i, 0, :K[0]] = torch.distributions.Multinomial(int(n), pi[i]).sample()
    # Propagate the counts
    for l in range(1, L):
        alpha = alpha_modules[l - 1](X[:, l - 1, :K[l - 1]])
        for parent in tree.getNodesAtDepth(l - 1 + selected_layers[0]):
            children = parent.children
            if len(children) <= 1:
                X[:, l, children[0].layer_index] = X[:, l - 1, parent.layer_index]
                continue
            children_index = [child.layer_index for child in children]
            alpha_children = alpha[:, children_index]
            for i in range(n_samples):
                parent_value = X[i, l - 1, parent.layer_index]
                # We add a bias to alpha to make sure alpha_j > 0
                prop = torch.distributions.Dirichlet(alpha_children[i] + 1e-8).sample()
                if parent_value != 0:
                    X[i, l, children_index] = torch.distributions.Multinomial(int(parent_value), prop).sample()
    return X
