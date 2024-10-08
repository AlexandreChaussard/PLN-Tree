import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from plntree.utils import viz, functions

from plntree.metrics import Shannon, InverseSimpson, Chao1, Simpson, Gini
from plntree.metrics import BrayCurtis, Jaccard, UniFrac
from plntree.metrics import MeanAbsoluteError, SpectralDistance, CosineSimilarity, GraphEditDistance


def alpha_metrics(taxonomy, layer):
    return {
        f'Shannon $\ell={layer}$': Shannon(taxonomy, layer),
        f'Simpson $\ell={layer}$': Simpson(taxonomy, layer),
        #f'Inverse Simpson $\ell={layer}$': InverseSimpson(taxonomy, layer),
        f'Chao1 $\ell={layer}$': Chao1(taxonomy, layer),
    }


def beta_metrics(taxonomy, layer):
    return {
        f'Bray Curtis $\ell={layer}$': BrayCurtis(taxonomy, layer),

        'Jaccard (unweighted)': Jaccard(taxonomy, weighted=False),
        'Jaccard (weighted)': Jaccard(taxonomy, weighted=True),

        'UniFrac (unweighted)': UniFrac(taxonomy, weighted=False),
        # Todo: weighted unifrac is not implementable atm (requires branch weights out of samples weights)
        # 'UniFrac (weighted)': UniFrac(taxonomy, weighted=True),
    }


def graph_metrics(taxonomy):
    return {
        'Mean Absoluted Error': MeanAbsoluteError(taxonomy),

        # Adjacency on cosine similarity is never showing interesting results
        # 'Cosine Similarity (adjacency, weighted)': CosineSimilarity(taxonomy, method="adjacency", binary=True),
        'Cosine Similarity (laplacian, weighted)': CosineSimilarity(taxonomy, method="laplacian", binary=True),
        'Cosine Similarity (unsigned_laplacian, weighted)': CosineSimilarity(taxonomy, method="unsigned_laplacian",
                                                                             binary=True),
        'Cosine Similarity (adjacency, unweighted)': CosineSimilarity(taxonomy, method="adjacency", binary=False),
        'Cosine Similarity (laplacian, unweighted)': CosineSimilarity(taxonomy, method="laplacian", binary=False),
        'Cosine Similarity (unsigned_laplacian, unweighted)': CosineSimilarity(taxonomy, method="unsigned_laplacian",
                                                                               binary=False),

        # Adjacency on spectral distance has proved to be more subject to cospectral situations than Laplacian
        # 'Spectral Distance (adjacency, weighted)': SpectralDistance(taxonomy, method="adjacency", binary=False),
        'Spectral Distance (laplacian, weighted)': SpectralDistance(taxonomy, method="laplacian", binary=False),
        'Spectral Distance (|laplacian|, weighted)': SpectralDistance(taxonomy, method="unsigned_laplacian",
                                                                      binary=False),
        # 'Spectral Distance (adjacency, unweighted)': SpectralDistance(taxonomy, method="adjacency", binary=True),
        'Spectral Distance (laplacian, unweighted)': SpectralDistance(taxonomy, method="laplacian", binary=True),
        'Spectral Distance (|laplacian|, unweighted)': SpectralDistance(taxonomy, method="unsigned_laplacian",
                                                                        binary=True),

        # Graph edit Distance hasn't proved to be efficient enough to be listed here
        # 'Graph Edit Distance': GraphEditDistance(taxonomy),
    }


def build_colors_groups(X_list, names=None):
    colors = []
    groups = []
    for i, X in enumerate(X_list):
        color = f"C{i}"
        if names is None:
            name = color
        else:
            name = names[i]
        colors += [color] * len(X)
        groups += [name] * len(X)
    return colors, groups


def plot_alpha_diversity(X_list, taxonomy, offset_layer=0, colors=None, groups_name=None, style='violin'):
    groups = []
    for i, X in enumerate(X_list):
        groups += [groups_name[i]] * len(X)

    def pad(X, n_pads):
        if n_pads == 0:
            return X
        return torch.nn.functional.pad(X, (0, 0, n_pads, 0, 0, 0), value=0)

    dataset = torch.cat([pad(X, offset_layer) for X in X_list], dim=0)
    K = [taxonomy.getNodesAtDepth(layer) for layer in range(taxonomy.depth)]
    for layer, K_l in enumerate(K):

        alpha = alpha_metrics(taxonomy, layer + offset_layer)
        fig, axs = plt.subplots(1, len(alpha), figsize=(17, 3))

        for i, name in enumerate(alpha.keys()):
            metric = alpha[name]
            values = metric.compute_batch(dataset)
            df = pd.DataFrame()
            df['Metric'] = values
            df['Group'] = groups
            nan_mask = pd.isna(df['Metric'])
            inf_mask = np.isinf(df['Metric'])
            df = df[~nan_mask & ~inf_mask]

            if style == 'violin':
                df = functions.remove_outliers_iqr(df, 'Metric', threshold=1.5)
                sns.violinplot(x='Group', y='Metric', data=df, ax=axs[i], palette=colors, cut=0., inner='boxplot')
            else:
                sns.boxplot(x='Group', y='Metric', data=df, ax=axs[i], showfliers=False, palette=colors)
            axs[i].set_title(name)

            # Rotate x-axis tick labels
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            # Show grid and set it below the boxplots
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            axs[i].set_axisbelow(True)

        fig.suptitle(f'Observed layers: [{offset_layer}, {layer + offset_layer}]', y=1.02)

    plt.show()


def plot_beta_diversity_similarity_graphs(X, taxonomy, layer, colors=None):
    beta = beta_metrics(taxonomy, layer)

    for name, metric in beta.items():
        df = pd.DataFrame(data=metric.compute_self_batch(X))
        viz.jupyter_threshold_update_graph(viz.weighted_network(1 - df), colors=colors, title=f"{name} - similarity")


def plot_beta_metrics_PCoA(X, taxonomy, layer, colors=None, labels=None, figsize=(12, 12)):
    beta = beta_metrics(taxonomy, layer)
    fig, axs = plt.subplots(len(beta), 1, figsize=figsize)

    for i, (name, metric) in enumerate(beta.items()):
        df = pd.DataFrame(data=metric.compute_self_batch(X))
        viz.PCoA_plot(df, title=f"{name} 2D dissimilarity", axs=axs[i], color=colors, labels=labels)


def plot_graph_metrics_similarity_graphs(X, taxonomy, colors=None):
    metrics = graph_metrics(taxonomy)

    for name, metric in metrics.items():
        df = pd.DataFrame(data=metric.compute_self_batch(X)).abs()
        normalization = (df.max() - df.min())[0]
        df_norm = df
        if normalization != 0:
            df_norm = (df - df.min()) / (df.max() - df.min())
        elif df.max()[0] != 0:
            df_norm = df / df.max()
        viz.jupyter_threshold_update_graph(viz.weighted_network(df_norm), colors=colors, title=f"Normalized {name}")


def plot_graph_metrics_PCoA(X, taxonomy, colors=None, labels=None, figsize=(12, 12)):
    graph = graph_metrics(taxonomy)
    fig, axs = plt.subplots(len(graph), 1, figsize=figsize)

    for i, (name, metric) in enumerate(graph.items()):
        df = pd.DataFrame(data=metric.compute_self_batch(X))
        viz.PCoA_plot(df, title=f"{name} - 2D", axs=axs[i], color=colors, labels=labels)


def vizualize_entities_abundance(data, layer, taxonomy, groups, figsize=(12, 8), title='Bacteria', data_layer_shift=0,
                                 hide_unique_children=False):
    df = pd.DataFrame()
    if hide_unique_children and layer + data_layer_shift >= 1:
        unique_children = []
        for node in taxonomy.getNodesAtDepth(layer + data_layer_shift - 1):
            if len(node.children) == 1:
                unique_children += [node.children[0].layer_index]
    ticks_colors = []
    curr_parent = None
    for node in taxonomy.getNodesAtDepth(layer + data_layer_shift):
        if hide_unique_children and node.layer_index in unique_children:
            continue
        layer_index = node.layer_index
        if node.parent != curr_parent:
            curr_parent = node.parent
        color = 'k'
        if curr_parent is not None:
            color = f"C{curr_parent.layer_index}"
        ticks_colors += [f"{color}"]
        for k, X in enumerate(data):
            entry = {}
            group = groups[k]
            entry['Group'] = group
            entry['Bacteria'] = node.name
            abundances = X[:, layer, layer_index].numpy()
            for j, value in enumerate(abundances):
                entry[f'sample_{j}'] = value
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    df_melted = pd.melt(df, id_vars=['Group', 'Bacteria'], var_name='Sample', value_name='Value')
    plt.figure(figsize=figsize)
    ax = sns.boxplot(x='Bacteria', y='Value', hue='Group', data=df_melted, fliersize=0.5, showfliers=False)
    [ax.axvline(x + .5, color='gray', linestyle='--', alpha=0.25) for x in ax.get_xticks()]
    plt.title(f'{title} distribution at layer {layer + data_layer_shift}')
    plt.xticks(rotation=-90)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(ticks_colors[i])

    return df

def plot_beta_diversity_PCoA(metrics, name="Beta diversity", colors=None, labels=None, figsize=(12, 12)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    df = pd.DataFrame(data=metrics)
    viz.PCoA_plot(df, title=f"{name} 2D dissimilarity", axs=axs, color=colors, labels=labels)

def plot_all_beta_diversity_PCoA(X_list, metrics_list, names, colors=None, labels=None, figsize=(12, 12)):
    fig, axs = plt.subplots(1, len(metrics_list), figsize=figsize)
    for i, dissimilarity in enumerate(metrics_list):
        if labels is not None:
            labels_samples = []
            colors_samples = []
            for j, X in enumerate(X_list):
                colors_samples += [colors[j]] * len(X)
                labels_samples += [labels[j]] * len(X)
            assert(len(colors_samples) == len(dissimilarity))
        else:
            colors_samples = None
            labels_samples = None
        df = pd.DataFrame(data=dissimilarity)
        if len(metrics_list) > 1:
            viz.PCoA_plot(df, title=f"{names[i]}", axs=axs[i], color=colors_samples, labels=labels_samples)
        else:
            viz.PCoA_plot(df, title=f"{names[i]}", axs=axs, color=colors_samples, labels=labels_samples)