import numpy as np
import matplotlib.pyplot as plt
import torch


# We define an object "Node" to build the abundance trees
class Node:

    # Constructor method, initializes the node by a recursive parent/children relationship
    # The value accounts for the abundance
    # The index and depth are indicating the position of the node in different annotations (matrix, vector, tree)
    def __init__(self, parent, children, value=1, index=0, depth=0, name="Unknown"):
        self.parent = parent

        self.children = children
        if self.children is None:
            self.children = []

        self.index = index
        self.depth = depth
        self.layer_index = 0
        self.graph_position = [0, depth]
        self.value = value
        self.name = name

    def hasParent(self):
        return self.parent is not None

    def addChild(self, child):
        self.children.append(child)

    def hasChildren(self):
        return self.countChildren() > 0

    def countChildren(self):
        return len(self.children)

    def countNonZeroChilren(self):
        count = 0
        for child in self.children:
            count += (child.is_zero()) * 1
        return count

    def getAbsoluteDegree(self):
        # The degree of a node in a tree corresponds to its count of children + its parent if it has one
        # Absolute in the sense that it doesn't account for whether or not the node is activated
        return len(self.children) + 1 * (self.parent is not None)

    def getEffectiveDegree(self):
        value = 0
        if self.parent is not None:
            value += self.parent.value
        for child in self.children:
            value += child.value
        return value

    def is_zero(self):
        return self.value == 0 or self.value == 0. or self.value < 1e-8


# We create a "Tree" object which is a composition of nodes, built from an adjacency matrix
class Tree:

    def __init__(self, adjacent_matrix):
        self.adjacent_matrix = np.array(adjacent_matrix)
        self.root = Node(None, None)
        self.depth = 0
        self.layers_nodes = {}

        self.nodes = np.array([Node(None, None, index=i) for i in range(len(self.adjacent_matrix))])
        self.nodes[0] = self.root

        current_index = 0
        while current_index < len(self.adjacent_matrix):
            # Fetch the current node
            current_node = self.nodes[current_index]
            # Look for the nodes that are connected to the current node, which are necessarily his children
            children_nodes_index = np.where(self.adjacent_matrix[current_index] != 0)[0]
            # For all child in the found children, associate their parent to the current node and append the child
            for child in self.nodes[children_nodes_index]:
                child.parent = current_node
                child.depth = current_node.depth + 1
                self.depth = max(child.depth, self.depth)
                current_node.addChild(child)
            current_index += 1

        def _internal_get_nodes_at_depth(depth):
            nodes_list = []
            for node_point in self.nodes:
                if node_point.depth == depth:
                    nodes_list.append(node_point)
            return nodes_list

        # Build the layer decomposition of the tree
        # Adjust the layer index of each node i.e. their position within their layer
        layer_index = 0
        layer = 0
        while True:
            nodes = _internal_get_nodes_at_depth(layer)
            if len(nodes) == 0:
                break
            for node in nodes:
                node.layer_index = layer_index
                layer_index += 1
            self.layers_nodes[layer] = nodes
            layer_index = 0
            layer += 1

    def getMaxDepth(self):
        max_depth = 0
        for node in self.nodes:
            if node.depth > max_depth:
                max_depth = node.depth
        return max_depth

    def getLeaves(self):
        return self.getNodesAtDepth(self.depth)

    def getNodesAtDepth(self, depth):
        return self.layers_nodes[depth]

    def getNode(self, depth, layer_index):
        for node in self.nodes:
            if node.depth == depth and node.layer_index == layer_index:
                return node
        return None

    def getLayersWidth(self):
        return {key: len(nodes) for key, nodes in self.layers_nodes.items()}

    def plot(self, space=10e10, title=None, fig=None, axs=None, cmap=plt.cm.get_cmap('Blues'), legend=True,
             threshold_abundance=10e-2):
        if fig is None or axs is None:
            fig, axs = plt.subplots(figsize=(15, 7))
        axs.set_yticks([])
        axs.set_xticks([])
        axs.axis('off')
        if title is None:
            axs.set_title("Tree representation")
        else:
            axs.set_title(title)

        if legend:
            # Create a ScalarMappable object with the colormap 'Blues'
            sm = plt.cm.ScalarMappable(cmap=cmap)

            # Set the limits of the colorbar to match your plot data
            sm.set_clim(0, 1)

            # Add a colorbar to the plot
            cbar = fig.colorbar(sm)
            cbar.set_label('Abundance (%)')

        graph_grid = {}
        index = 1
        for depth, nodes in self.layers_nodes.items():
            n_nodes = len(nodes)
            j = 0
            if n_nodes == 1:
                pos = [0]
            else:
                pos = np.linspace(
                    1 * n_nodes + space,
                    -1 * n_nodes - space,
                    n_nodes
                )
            while j < n_nodes:
                graph_grid[nodes[j].index] = [pos[j], -depth - 1]
                j += 1
                index += 1

        # Then recursively plot the nodes
        total_abundance = self.root.value

        def recursive_plot_nodes(node):
            if node.hasChildren():

                for i, child in enumerate(node.children):
                    if child.is_zero():
                        continue
                    child.graph_position = graph_grid[child.index]
                    axs.plot([child.graph_position[0]],
                             [child.graph_position[1]], color=cmap(child.value / total_abundance - 10e-10), marker="o",
                             markersize=12)
                    if child.value > threshold_abundance:
                        axs.text(
                            child.graph_position[0],
                            child.graph_position[1],
                            f'{child.index}',
                            ha='center',
                            va='center',
                            ma='center',
                            color='white',
                            fontsize='small'
                        )

                    recursive_plot_nodes(child)

        def recursive_plot_lines(node):
            if node.hasChildren():

                for i, child in enumerate(node.children):
                    if child.is_zero():
                        continue
                    child.graph_position = graph_grid[child.index]
                    axs.plot([child.graph_position[0], node.graph_position[0]],
                             [child.graph_position[1], node.graph_position[1]],
                             color=cmap(child.value / total_abundance - 10e-10), linestyle="-", marker="")

                    recursive_plot_lines(child)

        recursive_plot_lines(self.root)
        recursive_plot_nodes(self.root)

        # Adding the root to the graph
        axs.plot([self.root.graph_position[0]], [self.root.graph_position[1]], color=cmap(1 - 10e-10), marker="o",
                 markersize=12)
        axs.text(
            self.root.graph_position[0],
            self.root.graph_position[1],
            f'{self.root.index}',
            va='center', ha='center',
            color='white')

    def to_array(self, fill_value=0.):
        X = np.full((len(self.getLayersWidth()), np.max(list(self.getLayersWidth().values()))), fill_value=fill_value)
        for node in self.nodes:
            l = node.depth
            k = node.layer_index
            X[l][k] = node.value
        return X

    def to_adjacency_matrix(self, binary=False):
        G = np.zeros((len(self.nodes), len(self.nodes)))
        for node in self.nodes:
            for child in node.children:
                if binary:
                    G[node.index][child.index] = 1
                    G[child.index][node.index] = 1
                else:
                    G[node.index][child.index] = child.value
                    G[child.index][node.index] = node.value
        return G

    def absolute_degree_matrix(self):
        D = np.zeros((len(self.nodes), len(self.nodes)))
        for node in self.nodes:
            D[node.index][node.index] = node.getAbsoluteDegree()
        return D

    def effective_degree_matrix(self):
        D = np.zeros((len(self.nodes), len(self.nodes)))
        for node in self.nodes:
            D[node.index][node.index] = node.getEffectiveDegree()
        return D

    def hierarchical_compositional_constraint_fill(self, last_layers):
        K = list(self.getLayersWidth().values())
        K_max = max(K)
        L = len(K)
        L_obs = last_layers.shape[1]
        last_known_L = L - L_obs
        X = torch.zeros((len(last_layers), L, K_max))
        X[:, last_known_L:, :] = last_layers
        for l in range(last_known_L, 0, -1):
            for node in self.getNodesAtDepth(l):
                X[:, l - 1, node.parent.layer_index] += X[:, l, node.layer_index]
        return X

    def __str__(self):
        return f'Tree: {list(self.getLayersWidth().values())}'


# Abundance trees are a subclass of trees for which the value account for the abundance
class AbundanceTree(Tree):

    def __init__(self, adjancent_matrix, abundance_values):
        super().__init__(adjancent_matrix)
        self.abundance_values = abundance_values
        for node in self.nodes:
            value = (node.index == 0) * 1
            if node.index in abundance_values:
                value = abundance_values[node.index]
            node.value = value

    def __str__(self):
        return f'AbundanceTree: {list(self.getLayersWidth().values())}'


def tree_graph_builder(adjacency_matrix):
    """
    Builds a Tree graph object from its adjacency matrix
    Parameters
    ----------
    adjacency_matrix: square numpy array

    Returns
    -------
    Tree graph
    """
    return Tree(adjacency_matrix)


def abundance_tree_builder(tree_graph: Tree, abundance_matrix_per_layer, entities_name=None):
    """
    Builds a dataset of abundance trees or a single abundance tree based on the tree graph using the provided
    list of abundances sorted by layers.
    Parameters
    ----------
    tree_graph
    abundance_matrix_per_layer
    entities_name

    Returns
    -------

    """
    K_max = max(tree_graph.getLayersWidth().values())
    if abundance_matrix_per_layer.shape != (K_max, K_max):
        filled_matrix = np.zeros((K_max, K_max))
        for layer, K_l in tree_graph.getLayersWidth().items():
            if layer == len(abundance_matrix_per_layer):
                break
            filled_matrix[layer, :K_l] = abundance_matrix_per_layer[layer, :K_l]
        abundance_matrix_per_layer = filled_matrix

    adjacent_matrix = tree_graph.adjacent_matrix

    if type(abundance_matrix_per_layer) is not list and len(abundance_matrix_per_layer.shape) == 2:
        abundance_values = {}
        for node in tree_graph.nodes:
            l = node.depth
            k = node.layer_index
            x_kl = abundance_matrix_per_layer[l][k]
            abundance_values[node.index] = x_kl
            if entities_name is not None:
                node.name = entities_name[l][k]
    else:
        abundance_trees = []
        for matrix_per_layer in abundance_matrix_per_layer:
            abundance_trees.append(abundance_tree_builder(tree_graph, matrix_per_layer, entities_name))
        return abundance_trees

    return AbundanceTree(adjacent_matrix, abundance_values)


def partial_abundance_matrix_to_tree(tree_graph: Tree, abundance_matrix_per_layer, entities_name=None):
    K_max = max(list(tree_graph.getLayersWidth().values()))
    n_layers = len(tree_graph.getLayersWidth())
    X = np.zeros((n_layers, K_max))

    last_known_layer = n_layers - len(abundance_matrix_per_layer)
    X[last_known_layer:, :] = abundance_matrix_per_layer
    while last_known_layer > 0:
        children_nodes = tree_graph.getNodesAtDepth(last_known_layer)
        for child in children_nodes:
            parent_index = child.parent.layer_index
            child_index = child.layer_index
            X[last_known_layer - 1][parent_index] += X[last_known_layer][child_index]
        last_known_layer -= 1

    return abundance_tree_builder(tree_graph, X, entities_name)


def batch_partial_abundance_matrix_completion(tree, batch_abundance_matrix_per_layer):
    K_max = max(list(tree.getLayersWidth().values()))
    n_layers = len(tree.getLayersWidth())
    batch_size = batch_abundance_matrix_per_layer.size(0)
    X = torch.zeros((batch_size, n_layers, K_max))

    for batch_index, matrix in enumerate(batch_abundance_matrix_per_layer):
        abundance = partial_abundance_matrix_to_tree(tree, matrix.detach().numpy())
        X[batch_index, :, :] = torch.tensor(abundance.to_array())

    return X
