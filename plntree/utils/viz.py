import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx
from sklearn.manifold import MDS
from ipywidgets import interact, FloatSlider


def weighted_network(weight_matrix, names=None, colors=None):
    if names is None:
        names = {i: f'{i}' for i in range(len(weight_matrix))}
    if colors is None:
        colors = {i: 'blue' for i in range(len(weight_matrix))}

    sources = []
    targets = []
    weights = []

    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            if j <= i:
                continue
            sources.append(i)
            targets.append(j)
            weights.append(weight_matrix[i][j])

    sources = np.array(sources)
    targets = np.array(targets)
    weights = np.array(weights)

    G = nx.Graph()

    # Plot the nodes on the graph
    for i, k in enumerate(sources):
        G.add_node(names[k], size=15, title=names[k], color=colors[k], group=1)

    # Plot the edges on the graph
    for node, target, weight in zip(sources, targets, weights):
        if np.abs(weight) == 0:
            continue
        G.add_edge(str(node), str(target), weight=weight, value=weight, physics=True)

    return G


def jupyter_threshold_update_graph(G, colors=None, thresholds=(0., 1.), step=0.05, init_threshold=0., title="Graph"):
    if colors is None:
        colors = ['C0' for _ in G.nodes(data=True)]
    def update_graph(graph):
        return lambda threshold: update_threshold(threshold, graph)

    def update_threshold(threshold, graph):
        G_thresh = graph.copy()
        alpha = np.array([d['weight'] for u, v, d in graph.edges(data=True)])
        if len(alpha) > 0:
            alpha[np.isnan(alpha)] = 0

        for u, v, d in G.edges(data=True):
            if d['weight'] < threshold:
                G_thresh.remove_edge(u, v)
        fig, axs = plt.subplots()
        fig.suptitle(title)
        pos = nx.spring_layout(G_thresh)
        nx.draw_networkx_nodes(G_thresh, pos, ax=axs, node_size=20,
                               node_color=colors)
        if len(alpha > 0):
            nx.draw_networkx_edges(G_thresh, pos, ax=axs, alpha=alpha)
        else:
            nx.draw_networkx_edges(G_thresh, pos, ax=axs)
        plt.show()

    interact(update_graph(G),
             threshold=FloatSlider(min=thresholds[0], max=thresholds[1], step=step, init_threshold=init_threshold))


def plot_network(precision_matrix, labels=None, shape=(800, 800), physics=False, notebook=False):
    """
    Display the graph that corresponds to the precision matrix in entry.
    Labels can be applied to the nodes.

    Parameters
    ----------
    precision_matrix
    labels
    physics
    shape
    notebook

    Returns
    -------

    """

    sources = []
    targets = []
    covariance = []

    for i in range(precision_matrix.shape[0]):
        for j in range(precision_matrix.shape[1]):
            if j <= i:
                continue
            sources.append(i)
            targets.append(j)
            covariance.append(precision_matrix[i][j])

    sources = np.array(sources)
    targets = np.array(targets)
    covariance = np.array(covariance)

    def sign(value):
        return (value > 0) * 1

    def color_sign(value, color1="#75b1d9", color2="#f28963"):
        if sign(value) == 1:
            return color1
        else:
            return color2

    nx_graph = nx.Graph()

    # Plot the nodes on the graph
    for node in sources:
        nx_graph.add_node(str(node), size=15, title=node, group=1)

    # Plot the edges on the graph
    for node, target, weight in zip(sources, targets, covariance):
        if np.abs(weight) == 0:
            continue
        nx_graph.add_edge(str(node), str(target), weight=weight, value=np.abs(weight), color=color_sign(weight),
                          physics=physics)

    if notebook:
        nt = Network(f'{shape[0]}px', f'{shape[1]}px', notebook=notebook, cdn_resources='in_line')
    else:
        nt = Network(f'{shape[0]}px', f'{shape[1]}px', notebook=notebook)

    nt.from_nx(nx_graph)
    nt.show_buttons()

    # Add neighbour data to nodes on hover
    for node_obj in nt.nodes:
        node = int(node_obj["id"])
        if labels is not None:
            node_label = labels[node]
        else:
            node_label = str(node)
        node_obj["title"] = node_label

        interactions = ""
        for j in range(precision_matrix.shape[1]):
            if precision_matrix[node][j] != 0 and j != node:
                if labels is not None:
                    neighbour_label = labels[j]
                else:
                    neighbour_label = j
                interactions += f"- {neighbour_label}: {precision_matrix[node][j]}\n"
        if interactions == "":
            continue

        node_obj["title"] = str(node_label) + f"\nInteractions with: \n{interactions}"

    return nt


def PCoA_plot(dissimilarities, axs=None, title="Dissimilarities", color="C0", labels=None, n_runs=4):
    pcoa = MDS(n_components=2, metric=True, n_init=n_runs, dissimilarity='precomputed')
    pcoa_array = pcoa.fit_transform(dissimilarities)
    if axs is None:
        _, axs = plt.subplots()
    axs.set_title(f"MDS - {title}")
    if labels is None:
        axs.scatter(pcoa_array[:, 0], pcoa_array[:, 1], c=color)
    else:
        color = np.asarray(color)
        labels = np.asarray(labels)
        unique = np.unique(color)
        for c in unique:
            index = np.where(color == c)
            label = labels[index][0]
            axs.scatter(pcoa_array[index, 0], pcoa_array[index, 1], c=c, label=label, alpha=.4)
        axs.legend()
    return pcoa_array
