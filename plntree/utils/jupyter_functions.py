import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import ot

import plntree.metrics.viz as metrics_viz
from plntree.data import artificial_loader
from plntree.models import pln_lib
from plntree.utils import tree_utils
from plntree.utils import seed_all

def save_pkl(obj, prefix, name):
    folder = './experiments/saves'
    fileName = f'{folder}/{prefix}_{name}.pkl'
    print(f'Saving in file {fileName}')
    pickle.dump(obj, open(fileName, 'wb'))


def load_pkl(prefix, name):
    folder = './experiments/saves'
    fileName = f'{folder}/{prefix}_{name}.pkl'
    print(f'Loading file {fileName}')
    return pickle.load(open(fileName, 'rb'))


def savefig(name, extension='pdf', bbox_inches='tight'):
    plt.savefig(f"experiments/{name}.{extension}")


def generate_hierachical_tree(K, min_children=1, random_strategy=np.random.randn, seed=None):
    G = artificial_loader.generate_hierarchical_tree(K, min_children=min_children, random_strategy=random_strategy, seed=seed)
    return tree_utils.tree_graph_builder(G)


def vizualize_samples(dataloader, tree, selected_layers, n_viz=4, autofill=False, seed=None):
    seed_all(seed)
    fig, axs = plt.subplots(1, n_viz, figsize=(15, 4))
    legend = True
    for batch, sample in enumerate(dataloader):
        batch_X = sample[0]
        if autofill:
            X_autofill = tree.hierarchical_compositional_constraint_fill(batch_X)
        else:
            X_autofill = batch_X
        for X_i in X_autofill:
            if np.random.random() < 0.7:
                continue
            n_viz -= 1
            X_i = X_i.numpy()
            taxa_i = tree_utils.abundance_tree_builder(tree, X_i)
            taxa_i.plot(title=f"Total count {int(X_i[0][0])}", cmap=plt.cm.get_cmap('viridis_r'), fig=fig,
                        axs=axs[n_viz], legend=legend)
            if n_viz == 0:
                break
            legend = False
        break


def dataloader_to_tensors(dataloader):
    X_base, Z_base, O_base = torch.concatenate([X_batch for (X_batch, Z_batch, O_batch) in dataloader],
                                               dim=0), torch.concatenate(
        [Z_batch for (X_batch, Z_batch, O_batch) in dataloader], dim=0), torch.concatenate(
        [O_batch for (X_batch, Z_batch, O_batch) in dataloader], dim=0)
    return X_base, Z_base, O_base


def df_entities_distributions(data, layer, taxonomy, groups, figsize=(12, 8), title='Bacteria', data_layer_shift=0,
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
    return df_melted, ticks_colors


def vizualize_entities_distributions(model, entries_list, groups, title='Abundance', figsize=(12, 8)):
    fig, axs = plt.subplots(len(model.K), 1, figsize=figsize)
    for layer in range(len(model.K)):
        data_layer_shift = model.selected_layers[0]
        df_melted, ticks_colors = df_entities_distributions(entries_list, layer, model.tree, groups,
                                                            figsize=figsize, title=title,
                                                            data_layer_shift=model.selected_layers[0])
        sns.boxplot(x='Bacteria', y='Value', hue='Group', data=df_melted, fliersize=0.5, showfliers=False,
                    ax=axs[layer])
        [axs[layer].axvline(x + .5, color='gray', linestyle='--', alpha=0.25) for x in axs[layer].get_xticks()]
        axs[layer].set_title(f'{title} distribution at layer {layer + data_layer_shift}')
        if layer < len(model.K) - 1:
            axs[layer].set_xlabel('')
        else:
            axs[layer].set_xlabel('Entity')
        plt.xticks(rotation=-90)
        for i, tick in enumerate(axs[layer].get_xticklabels()):
            tick.set_color(ticks_colors[i])
    plt.subplots_adjust(hspace=0.16)


def multinomial_probas(Z, model):
    probas = torch.zeros_like(Z)
    for node in model.tree.nodes:
        if node.depth < model.selected_layers[0]:
            continue
        layer = node.depth + 1 - model.selected_layers[0]
        if layer >= Z.size(1):
            break
        children = node.children
        if node.hasChildren():
            children_index = [child.layer_index for child in children]
            probas[:, layer, children_index] = torch.softmax(Z[:, layer, children_index], dim=-1)
    probas[:, 0, :] = Z[:, 0, :]
    return probas.detach()


def to_proportion(X, K):
    normalized = X.clone()
    for layer, K_l in enumerate(K):
        normalized[:, layer, :K_l] = normalized[:, layer, :K_l] / normalized[:, layer, :K_l].sum(dim=-1, keepdims=True)
    return normalized


def mae_tree(X1, X2, K):
    m = 0
    for layer, K_l in enumerate(K):
        m += (X1[:, layer, :K_l] - X2[:, layer, :K_l]).abs().mean(dim=-1)
    m /= layer
    return m.mean(dim=0)


from plntree.utils.model_utils import Vect1OrthogonalProjectorHierarchical


def identifiable_projector(model, Z):
    for layer, K_eff in enumerate(model.effective_K):
        projector = Vect1OrthogonalProjectorHierarchical(model.tree, layer + model.selected_layers[0], K_eff)
        Z[:, layer, model.layer_masks[layer]] = projector(Z[:, layer, model.layer_masks[layer]])

    # Check that the projection is in the right space (should all be equal to 0)
    for node in model.tree.nodes:
        if node.depth < model.selected_layers[0]:
            continue
        children = node.children
        if len(children) < 1:
            continue
        children_index = [child.layer_index for child in children]
        layer = node.depth - model.selected_layers[0]
        assert (Z[:, layer + 1, children_index].sum(axis=-1).mean() < 1e-8)
    return Z


def learn_pln(X_base, K, seed=None):
    seed_all(seed)
    pln_layers = []
    for layer, K_l in enumerate(K):
        pln = pln_lib.fit(X_base, layer=layer, K=K, tol=1e-8)
        pln_layers.append(pln)
    for layer in range(len(K)):
        pln_layers[layer].show()
    return pln_layers


def generate_pln_data(pln_layers, n_samples, K, selected_layers, X_base, tree, seed=None):
    n_samples = 10_000

    X_pln = torch.zeros(n_samples, len(K), max(K))
    Z_pln = torch.zeros(n_samples, len(K), max(K))
    X_pln_enc = torch.zeros(len(X_base), len(K), max(K))
    Z_pln_enc = torch.zeros(len(X_base), len(K), max(K))

    for layer, K_l in enumerate(K):
        pln = pln_layers[layer]
        X_pln_numpy, Z_pln_numpy = pln_lib.sample(pln, n_samples, seed=seed)
        Z_pln_enc_numpy = pln_lib.encode(pln, seed=seed)
        X_pln_enc_numpy = pln_lib.decode(Z_pln_enc_numpy, seed=seed)

        X_pln[:, layer, :K_l] = torch.tensor(X_pln_numpy)
        Z_pln[:, layer, :K_l] = torch.tensor(Z_pln_numpy)
        Z_pln_enc[:, layer, :K_l] = torch.tensor(Z_pln_enc_numpy)
        X_pln_enc[:, layer, :K_l] = torch.tensor(X_pln_enc_numpy)

    X_pln_fill = X_pln[:, -1, :K_l].unsqueeze(1)
    X_pln_enc_fill = X_pln_enc[:, -1, :K_l].unsqueeze(1)
    X_pln_fill = tree.hierarchical_compositional_constraint_fill(X_pln_fill)[:, selected_layers[0]:]
    X_pln_enc_fill = tree.hierarchical_compositional_constraint_fill(X_pln_enc_fill)[:, selected_layers[0]:]

    return X_pln, Z_pln, X_pln_fill, X_pln_enc, Z_pln_enc, X_pln_enc_fill


def learn_multiple_models(n_repeat, learn_model, save_as=None):
    estimators = []
    for _ in range(n_repeat):
        print('Learning model:', _ + 1, '/', n_repeat)
        estimators.append(learn_model())
        if save_as is not None:
            save_pkl(estimators, *save_as)
    return estimators


def generate_multiple_models(models, n_samples, X_base, seed=None):
    generations = []
    encodings = []
    for estimator, losses in models:
        X_plntree, Z_plntree, O_plntree = estimator.sample(n_samples, seed=seed)
        Z_plntree_enc, O_plntree_enc = estimator.encode(X_base, seed=seed)
        X_plntree_enc = estimator.decode(Z_plntree_enc, O_plntree_enc, seed=seed)
        generations.append((X_plntree, Z_plntree, O_plntree))
        encodings.append((X_plntree_enc, Z_plntree_enc, O_plntree_enc))
    return generations, encodings


def concat_multiple_generations(generations, limit_per_model=None):
    X, Z, O = [], [], []
    for x, z, o in generations:
        if limit_per_model is None:
            X.append(x)
            Z.append(z)
            O.append(o)
        else:
            X.append(x[:limit_per_model])
            Z.append(z[:limit_per_model])
            O.append(o[:limit_per_model])
    return torch.cat(X), torch.cat(Z), torch.cat(O)


def add_offset(Z, O):
    Z_offset = Z[:, 0] + O
    Z_offset = torch.cat((Z_offset.unsqueeze(1), Z[:, 1:]), dim=1)
    return Z_offset


def plot_alpha_diversity(X_list, taxonomy, offset_layer=0, colors=None, groups_name=None, style='violin', saveName=''):
    groups = []
    for i, X in enumerate(X_list):
        groups += [groups_name[i]] * len(X)

    def pad(X, n_pads):
        if n_pads == 0:
            return X
        return torch.nn.functional.pad(X, (0, 0, n_pads, 0, 0, 0), value=0)

    dataset = torch.cat([pad(X, offset_layer) for X in X_list], dim=0)
    K = list(taxonomy.getLayersWidth().values())[offset_layer:]
    alpha = metrics_viz.alpha_metrics(taxonomy, 0)
    fig, axs = plt.subplots(len(alpha), len(K), figsize=(15, 8))
    for layer, K_l in enumerate(K):

        alpha = metrics_viz.alpha_metrics(taxonomy, layer + offset_layer)

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
                df = remove_outliers_iqr(df, 'Metric', threshold=1.5)
                sns.violinplot(x='Group', y='Metric', data=df, ax=axs[i][layer], palette=colors, cut=0.,
                               inner='boxplot')
            else:
                sns.boxplot(x='Group', y='Metric', data=df, ax=axs[i][layer], showfliers=False, palette=colors)
            axs[i][layer].set_title(name)
            axs[i][layer].set_xlabel('')

            # Rotate x-axis tick labels
            if i == len(alpha) - 1:
                axs[i][layer].set_xticklabels(axs[i][layer].get_xticklabels(), rotation=45)
            else:
                axs[i][layer].set_xticklabels([], rotation=45)
            # Show gridaxs[i][layer].set_xticklabels(axs[i][layer].get_xticklabels(), rotation=45) and set it below the boxplots
            axs[i][layer].grid(True, which='both', linestyle='--', linewidth=0.5)
            axs[i][layer].set_axisbelow(True)
    if len(saveName) > 0:
        savefig(f'{saveName}_alpha_diversity')


def correlation(X_base, X_comp, model):
    correlations = []
    for layer, K_l in enumerate(model.K):
        mask = model.layer_masks[layer]
        corr = torch.zeros(len(X_base))
        for i in range(len(X_base)):
            X_base_l = X_base[i, layer, mask]
            X_comp_l = X_comp[i, layer, mask]
            stack = torch.stack((X_base_l, X_comp_l))
            corr[i] = torch.corrcoef(stack)[0][1]
        correlations.append(corr)
    return correlations


def correlation_3d_plot(X_base, X_list, groups, model, bins=30, hist_lim=1000, saveName=''):
    # Create subplots
    K = model.K
    fig, axs = plt.subplots(1, len(K), figsize=(15, 5), subplot_kw={'projection': '3d'})

    correlations_list = [correlation(X_base, X_comp, model) for X_comp in X_list]
    viridis = matplotlib.colormaps.get_cmap('viridis')
    colors = [viridis(1e-8 + (i + 1) / len(correlations_list)) for i in range(len(correlations_list))]

    # Plot 3D histograms for each list of tensors
    for i in range(len(axs)):

        for j, (color, label) in enumerate(zip(colors, groups)):
            hist, bins = np.histogram(correlations_list[j][i], bins=bins, density=True)
            hist[hist > hist_lim] = hist_lim
            xs = (bins[:-1] + bins[1:]) / 2
            axs[i].bar(xs, hist, zs=j, zdir='y', width=0.03, color=color, alpha=0.5)

        axs[i].set_ylabel('')
        axs[i].set_xlabel('Correlation')
        axs[i].set_zlabel('Frequency')
        axs[i].set_title(f'$\ell={i + 1}$')
    for ax in axs:
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(groups, rotation=-75)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10)

    if len(saveName) > 0:
        savefig(f"{saveName}_3d_correlations")
    plt.show()


def mae(X_base, X_comp):
    n_models = X_comp.size(0) // X_base.size(0)
    n_samples = X_base.size(0)
    m = torch.zeros(n_models)
    for i in range(n_models):
        m[i] = torch.mean((torch.flatten(X_base - X_comp[i * n_samples:(i + 1) * n_samples])).abs())
    return m


from scipy.stats import wasserstein_distance, ks_2samp, entropy, gaussian_kde


def rank_df(df):
    def rank_columns(row):
        return sorted(df.columns, key=lambda col: row[col], reverse=False)

    ranked_df = pd.DataFrame(df.apply(rank_columns, axis=1).tolist(), index=df.index)
    return ranked_df


def rank_models(dist_df, filtered_index=[]):
    ranked_df = rank_df(dist_df)
    for entry in filtered_index:
        ranked_df = ranked_df[~ranked_df.index.str.contains(entry)]
    models_id = dist_df.columns
    models_ranks = {}
    for identifier in models_id:
        models_ranks[identifier] = []
    for col in ranked_df.columns:
        for identifier in ranked_df[col]:
            models_ranks[identifier].append(col)
    ranking_df = pd.DataFrame(models_ranks)
    df = pd.DataFrame(columns=models_id)
    df = df.append(ranking_df.mean(axis=0), ignore_index=True)
    df = df.append(ranking_df.std(axis=0), ignore_index=True)
    df.index = ['Mean rank', 'Std rank']
    df = df.sort_values(by=df.index[0], axis=1)
    return df


def kl_divergence(p_samples, q_samples, bias=1e-15):
    try:
        kde_p = gaussian_kde(p_samples)
        kde_q = gaussian_kde(q_samples)
    except:
        return np.inf
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    grid = np.linspace(min_val, max_val, 1000)
    pdf_p = kde_p(grid) + bias
    pdf_q = kde_q(grid)
    pdf_q[pdf_q < bias] = bias
    kl_divergence = entropy(pdf_p, pdf_q)
    if np.isnan(np.sum(pdf_p * np.log(pdf_p / pdf_q + bias))):
        print(pdf_p / pdf_q)
    return entropy(pdf_p, pdf_q)


def total_variation(samples1, samples2):
    try:
        kde_1 = gaussian_kde(samples1)
        kde_2 = gaussian_kde(samples2)
    except:
        return np.inf
    min_val = min(np.min(samples1), np.min(samples2))
    max_val = max(np.max(samples1), np.max(samples2))
    grid = np.linspace(min_val, max_val, 1000)
    pdf_1 = kde_1(grid)
    pdf_2 = kde_2(grid)
    return 0.5 * np.sum(np.abs(pdf_1 - pdf_2) / len(grid))


def kolmogorov_smirnov(samples1, samples2):
    ks_statistic, _ = ks_2samp(samples1, samples2)
    return ks_statistic


def compute_alpha_diversity(X_list, taxonomy, groups_name, offset_layer):
    groups = []
    for i, X in enumerate(X_list):
        groups += [groups_name[i]] * len(X)

    def pad(X, n_pads):
        if n_pads == 0:
            return X
        return torch.nn.functional.pad(X, (0, 0, n_pads, 0, 0, 0), value=0)

    dataset = torch.cat([pad(X, offset_layer) for X in X_list], dim=0)
    K = list(taxonomy.getLayersWidth().values())[offset_layer:]
    alpha_df_list = [pd.DataFrame() for _ in range(len(metrics_viz.alpha_metrics(taxonomy, offset_layer)))]
    for layer, K_l in enumerate(K):
        alpha = metrics_viz.alpha_metrics(taxonomy, layer + offset_layer)
        for i, name in enumerate(alpha.keys()):
            metric = alpha[name]
            values = metric.compute_batch(dataset)
            alpha_df_list[i][f'{name}'] = values
            #nan_mask = pd.isna(alpha_df_list[i][f'{name}'])
            #inf_mask = np.isinf(alpha_df_list[i][f'{name}'])
            #alpha_df_list[i] = alpha_df_list[i][~nan_mask & ~inf_mask]
    for df in alpha_df_list:
        df[f'Group'] = groups
    return alpha_df_list


def compute_alpha_distance(X_list, taxonomy, groups, offset_layer, distance=wasserstein_distance,
                           filtered_index=['Chao1'], order=False):
    alpha_df_list = compute_alpha_diversity(X_list, taxonomy, groups_name=groups, offset_layer=offset_layer)
    results = pd.DataFrame(columns=groups[1:])
    for df in alpha_df_list:
        for metric in df.columns[:-1]:
            res = []
            metric_base = df[df['Group'] == 'Data'][metric]
            metric_base = metric_base[~pd.isna(metric_base) & ~np.isinf(metric_base)]
            groups_ = df['Group'].unique()
            for g in groups_:
                if g == 'Data':
                    continue
                metric_vs = df[df['Group'] == g][metric]
                # Filter out nan
                metric_vs = metric_vs[~pd.isna(metric_vs) & ~np.isinf(metric_vs)]
                d = distance(metric_base, metric_vs)
                res.append(d)
            index = results.index
            results = results.append(dict(zip(groups[1:], res)), ignore_index=True)
            results.index = list(index) + [metric]
    ranks = rank_models(results, filtered_index=filtered_index)
    if order:
        return bold_min(pd.concat((ranks, results)))
    return bold_min(pd.concat((results, ranks)))


def compute_distribution_distance(X_base, X_comp, n_split, comp_names, K, distance, order=False):
    data = torch.zeros(X_base.shape[1], len(X_comp), n_split)
    for k in range(n_split):
        df = pd.DataFrame(columns=comp_names)
        for l, K_l in enumerate(K):
            row = []
            for i, name in enumerate(comp_names):
                X_c = X_comp[i]
                n_samples = X_c.shape[0] // n_split
                X_c = X_c[k * n_samples:(k + 1) * n_samples]
                row.append(distance(X_base[:, l, :K_l].numpy(), X_c[:, l, :K_l].numpy()))
            index = df.index
            df = df.append(dict(zip(comp_names, row)), ignore_index=True)
            df.index = list(index) + [f'$\ell = {l}$']
        data[:, :, k] = torch.tensor(df.to_numpy())
    means = pd.DataFrame(index=df.index, columns=df.columns, data=data.mean(dim=-1))
    stds = pd.DataFrame(index=df.index, columns=df.columns, data=data.std(dim=-1))
    for i in range(means.shape[0]):
        if i >= means.shape[0]:
            continue
        for j in range(means.shape[1]):
            means.iloc[i, j] = f"{np.round(means.iloc[i, j], 4)} ({np.round(stds.iloc[i, j], 4)})"
    ranks = rank_models(means, filtered_index=[])
    if order:
        styled_means = bold_min(pd.concat((ranks, means)))
    else:
        styled_means = bold_min(pd.concat((means, ranks)))
    return styled_means


def emd(batch1, batch2):
    # Multivariate Wasserstein distance
    M = ot.dist(batch1, batch2, metric='euclidean')
    a_weight, b_weight = np.ones(batch1.shape[0], ) / batch1.shape[0], np.ones(batch2.shape[0], ) / batch2.shape[0]
    return ot.emd2(a_weight, b_weight, M)


def bold_min(df):
    def highlight_min(data):
        attr = 'font-weight: bold'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_min = data == data.min()
            return [attr if v else '' for v in is_min]
        else:  # DataFrame from .apply(axis=None)
            is_min = data == data.min()
            return pd.DataFrame(np.where(is_min, attr, ''),
                                index=data.index, columns=data.columns)

    return df.style.apply(highlight_min, axis=1)


def repeated_metric_compute(tree, X_base, X_comp, groups, n_split, distance, offset_layer, order=False,
                            filtered_index=('Chao1',)):
    viridis = matplotlib.colormaps.get_cmap('viridis')
    means = pd.DataFrame()
    stds = pd.DataFrame()
    for group, X_c in zip(groups, X_comp):
        n_samples = X_c.shape[0] // n_split
        X_list = [X_base]
        groups = ['Data']
        colors = [viridis(0.3)]
        for k in range(n_split):
            X_list += [X_c[k * n_samples:(k + 1) * n_samples]]
            groups += [f'R{k}']
            colors += [viridis(0.3 + (0.69 * (k + 1) / n_split))]
        df_metric = compute_alpha_distance(X_list, tree, groups, offset_layer=offset_layer, distance=distance)
        means[group] = df_metric.data.mean(axis=1)[:-2]
        stds[group] = df_metric.data.std(axis=1)[:-2]
    for i in range(means.shape[0]):
        if i >= means.shape[0]:
            continue
        for j in range(means.shape[1]):
            means.iloc[i, j] = f"{np.round(means.iloc[i, j], 4)} ({np.round(stds.iloc[i, j], 4)})"
    ranks = rank_models(means, filtered_index=filtered_index)
    if order:
        styled_means = bold_min(pd.concat((ranks, means)))
    else:
        styled_means = bold_min(pd.concat((means, ranks)))
    return styled_means