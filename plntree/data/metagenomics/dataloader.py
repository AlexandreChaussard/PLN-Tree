import numpy as np
import torch
import pandas as pd
from plntree.utils.tree_utils import Tree
from plntree.utils import seed_all
import warnings
import os

warnings.filterwarnings("ignore")


def load_raw_data(directory='./'):
    """
    Load the raw data from the path
    Outputs two dataframes: one for the raw abundance and one for the metadata
    The raw abundances are represented by percentage of the total host microbiome composition
    """
    data = pd.read_csv(os.path.join(directory, 'abundance.txt'), sep='\t', header=None)
    data.columns = ['Feature'] + [f'P{i}' for i in range(1, len(data.columns))]
    metadata = data.head(210)
    raw_abundance = data.tail(len(data) - 211)
    metadata.set_index('Feature', inplace=True)
    return raw_abundance, metadata.T


def raw_abundance_filter(raw_abundance, precision=4):
    """
    Filter the raw abundance data to a certain precision level
    Parameters
    ----------
    raw_abundance
    precision

    Returns
    -------
    filtered_abundance
    """
    levels = ['k', 'p', 'c', 'o', 'f', 'g', 's']
    if precision > len(levels) or precision < 2:
        raise ValueError(f"Precision should be less than {len(levels)} and superior to 2.")

    selected_levels = levels[:precision]
    filtered_out_levels = levels[precision:]

    # Filter out the levels
    filtered_abundance = raw_abundance.copy()
    for level in filtered_out_levels:
        filtered_abundance = filtered_abundance[~filtered_abundance['Feature'].str.contains('\|' + level + '__')]
    max_level = selected_levels[-1]
    filtered_abundance = filtered_abundance[filtered_abundance['Feature'].str.contains('\|' + max_level + '__')]
    filtered_abundance.set_index('Feature', inplace=True)
    filtered_abundance = filtered_abundance.astype('float')
    # Normalize the abundances to proportions
    filtered_abundance = filtered_abundance / filtered_abundance.sum(axis=0)
    # If we have divided by zero, we replace the NaN values by 0
    filtered_abundance[filtered_abundance.isna()] = 0.
    return filtered_abundance


def filter_diseases(raw_abundances, metadata, diseases):
    """
    Filter the raw abundances and metadata to keep only the patients with the specified diseases
    Parameters
    ----------
    raw_abundances
    metadata
    diseases

    Returns
    -------
    filtered_abundances, filtered_metadata
    """
    metadata_filtered = metadata[metadata['disease'].isin(diseases)]
    columns = ['Feature'] + list(metadata_filtered.index)
    return raw_abundances[columns], metadata_filtered


def get_taxonomy(abundance_df):
    """
    Get the taxonomy tree from the abundance dataframe
    Parameters
    ----------
    abundance_df

    Returns
    -------
    Tree
    """
    levels = ['k', 'p', 'c', 'o', 'f', 'g', 's']
    selected_levels = []
    abundances = abundance_df.copy()
    abundances = abundances.reset_index()
    for level in levels:
        if abundances['Feature'].str.contains(level + '__').any():
            selected_levels.append(level)
    # Building the taxonomic tree
    mapping_df = abundances['Feature'].str.split('|', expand=True)
    mapping_df.columns = selected_levels
    # Build a dictionary to map the parent to the children
    mapping_parent = {}
    for l in range(-1, len(selected_levels)):
        if l == -1:
            mapping_parent['root'] = mapping_df[selected_levels[0]].unique().astype('str')
            continue
        level = selected_levels[l]
        if l < len(selected_levels) - 1:
            next_level = selected_levels[l + 1]
        taxa = mapping_df[level].unique()
        for taxon in taxa:
            if l < len(selected_levels) - 1:
                children = mapping_df[mapping_df[level] == taxon][next_level].unique().astype('str')
            else:
                children = []
            mapping_parent[taxon] = children
    # Build a dictionary to map each node to an index
    mapping_index = {}

    # We do it in a recursive fashion for visual clarity of the tree
    def recursive_indexing(parent):
        children = mapping_parent[parent]
        for child in children:
            mapping_index[child] = len(mapping_index)
            recursive_indexing(child)

    mapping_index['root'] = 0
    recursive_indexing('root')
    # Mapping the parent index to the children index,
    # equivalent to mapping_parent dictionnary but ready to turn into an adjacent matrix
    parent_children_index = {}
    for parent in mapping_parent.keys():
        parent_index = mapping_index[parent]
        parent_children_index[parent_index] = [mapping_index[child] for child in mapping_parent[parent]]
    # We can now build the adjacent matrix
    size = len(parent_children_index)
    adjacent_matrix = np.zeros((size, size))
    for parent_index in parent_children_index.keys():
        children_index = parent_children_index[parent_index]
        adjacent_matrix[parent_index][children_index] = 1

    # Now we can build the global tree architecture
    taxonomy = Tree(adjacent_matrix=adjacent_matrix)
    # We apply the name of the bacteria to each node in the tree using the matching dictionnary
    for node in taxonomy.nodes:
        for name, index in mapping_index.items():
            if index == node.index:
                node.name = name
                break

    return taxonomy


def hierarchical_dataset(abundance_df, taxonomy, offset_layer):
    """
    Build the hierarchical dataset from the abundance dataframe
    Parameters
    ----------
    abundance_df
    taxonomy
    offset_layer

    Returns
    -------

    """
    K = list(taxonomy.getLayersWidth().values())[offset_layer:]
    K_max = max(K)
    L = len(K)
    X = torch.zeros((len(abundance_df.T), L, K_max))
    patient_ids = abundance_df.columns
    for i, patient in enumerate(patient_ids):
        for leaf in taxonomy.getNodesAtDepth(taxonomy.depth):
            full_name = leaf.name
            parent = leaf.parent
            while parent is not None:
                if parent.name == 'root':
                    break
                full_name = parent.name + '|' + full_name
                parent = parent.parent
            X[i, L - 1, leaf.layer_index] = abundance_df.loc[full_name, patient]
    for depth in range(taxonomy.depth, offset_layer, -1):
        l = depth - offset_layer
        for node in taxonomy.getNodesAtDepth(depth):
            parent = node.parent
            X[:, l - 1, parent.layer_index] += X[:, l, node.layer_index]
    return X, patient_ids


def rarefy(abundance_df, offset, seed=None):
    """
    Rarefy the abundance dataframe
    Parameters
    ----------
    abundance_df

    Returns
    -------
    rarefied_df
    """
    seed_all(seed)
    N = np.exp(offset)
    rarefied_df = abundance_df.copy()
    # Normalize the abundances (in case it's not already been done)
    rarefied_df = rarefied_df / rarefied_df.sum(axis=0)
    # If we have divided by zero, we replace the NaN values by 0
    rarefied_df[rarefied_df.isna()] = 0.
    rarefied_df = rarefied_df.apply(lambda x: np.random.multinomial(N, x), axis=0)
    return rarefied_df


def prevalence_filter(abundance_df, threshold=1e-6):
    """
    Filter the abundance dataframe based on the prevalence of the bacteria
    Parameters
    ----------
    abundance_df
    threshold

    Returns
    -------
    filtered_df
    """
    filtered_df = abundance_df.copy()
    # Normalize the abundances (in case it's not already been done)
    filtered_df = filtered_df / filtered_df.sum(axis=0)
    # If we have divided by zero, we replace the NaN values by 0
    filtered_df[filtered_df.isna()] = 0.
    # Measure the prevalence of the bacteria
    empirical_proba = filtered_df.mean(axis=1)
    filtered_df = filtered_df.loc[empirical_proba > threshold]
    return filtered_df


def exclude_taxon(abundance_df, taxon_name):
    """
    Exclude a taxon from the abundance dataframe
    Parameters
    ----------
    abundance_df
    taxon_name

    Returns
    -------
    filtered_df
    """
    filtered_df = abundance_df.copy()
    filtered_df = filtered_df[~filtered_df.index.str.contains(taxon_name)]
    return filtered_df
