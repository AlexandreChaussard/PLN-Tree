from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import exp, sum, logsumexp
from torch.distributions.poisson import Poisson

from plntree.models import BaseModel
from plntree.utils.model_utils import batch_trace, BoundLayer, offsets, SimplexParameter, \
    sample_gaussian_mixture, density_gaussian_mixture
from plntree.utils.tree_utils import partial_abundance_matrix_to_tree
from plntree.utils.variational_approximations import VariationalApproximation, mean_field


class _PLNTree(ABC, BaseModel):

    def __init__(
            self,
            tree,
            selected_layers=None,
            variance_bounds=(1e-8, 10.),
            mean_bounds=(-12., 12.),
            diagonal_model=False,
            use_smart_init=True,
            variational_approx="mean_field",
            normalize=True,
            offset_method="zeros",
            variational_approx_params=None,
            pln_layer=0,
    ):
        BaseModel.__init__(self, False)
        self.tree = tree
        self.diagonal_model = diagonal_model
        # Decide at which layer the PLN model is applied
        self.pln_layer = pln_layer

        # Parse the offset method
        if "gmm" in offset_method:
            self.offset_gmm_components = int(offset_method.split("_")[1])
            if self.offset_gmm_components < 1:
                raise ValueError("The number of components for the GMM offset model must be at least 1.")
            elif self.offset_gmm_components == 1:
                offset_method = "gaussian"
            else:
                offset_method = "gmm"
        assert (offset_method in ["zeros", "constant", "logsum", "gaussian", "gmm"])
        self.offset_method = offset_method

        # Define the offset model if enabled
        if offset_method == "constant":
            self.offset_constant = nn.Parameter(3 + torch.zeros(1))
        if offset_method == "gaussian":
            self.offset_mean = nn.Parameter(3 + torch.zeros(1))
            self.offset_log_var = nn.Parameter(torch.zeros(1))
            self.offset_m_fun = nn.Sequential(
                nn.Linear(1, 1),
                BoundLayer(-12., 12., smoothing_factor=0.2)
            )
            self.offset_log_s_fun = nn.Sequential(
                nn.Linear(1, 1),
                BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
            )
        elif offset_method == "gmm":
            self.offset_mixture_weights = SimplexParameter(self.offset_gmm_components)
            self.offset_mixture_means = [
                nn.Parameter(3 + torch.zeros(1))
                for _ in range(self.offset_gmm_components)
            ]
            self.offset_mixture_log_vars = [
                nn.Parameter(torch.zeros(1))
                for _ in range(self.offset_gmm_components)
            ]
            self.offset_mixture_m_fun = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, 1),
                    BoundLayer(-12., 12., smoothing_factor=0.2)
                )
                for _ in range(self.offset_gmm_components)
            ])
            self.offset_mixture_log_s_fun = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, 1),
                    BoundLayer(np.log(1e-8), np.log(10.), smoothing_factor=0.1)
                )
                for _ in range(self.offset_gmm_components)
            ])

        min_layer = 0
        if selected_layers is None:
            selected_layers = [min_layer, len(self.tree.getLayersWidth()) - 1]
        if selected_layers[-1] < 0:
            selected_layers[-1] = len(self.tree.getLayersWidth()) + selected_layers[-1]
        # Among the selected layers, we skip the root in the model
        selected_layers[0] = max(selected_layers[0], min_layer)
        self.selected_layers = selected_layers
        # The number of nodes per selected layer
        self.K = list(self.tree.getLayersWidth().values())[selected_layers[0]:selected_layers[1] + 1]
        # The layers contain nodes that might be unique children, in which case they are modelled by Dirac
        # distributions. Hence we need to filter them out in the Poisson modelisation.
        # The first layer does not have unique children, so we can already fill it with True masks.
        K_max = max(self.K)
        self.layer_masks = [[True] * self.K[0] + [False] * (K_max - self.K[0])]
        # We can also compute the effective K values, which account for the number of modelled nodes using Gaussians
        # at each layer.
        self.effective_K = [self.K[0]]
        for layer, K_l in enumerate(self.K):
            # We've reached the last layer, so we can stop
            if layer == len(self.K) - 1:
                break
            mask = []
            effective_K = 0
            for node in self.tree.getNodesAtDepth(layer + self.selected_layers[0]):
                if len(node.children) == 1:
                    mask += [False]
                else:
                    mask += [True] * len(node.children)
                    effective_K += len(node.children)
            while len(mask) < K_max:
                mask += [False]
            self.layer_masks += [mask]
            self.effective_K += [effective_K]

        # Whether to normalize the means of the latents
        self.normalize = normalize

        # Whether to use a warm start for the parameters
        self.use_smart_init = use_smart_init

        # Limit the outputs of the networks
        self.means_bounds = mean_bounds
        self.variance_bounds = variance_bounds

        # Choose the variational approximation
        self.variational_approx = VariationalApproximation(variational_approx)
        # If the variational approximation is mean field, it only depends on the X^l
        if self.variational_approx == VariationalApproximation.MEAN_FIELD:
            proportion = offset_method != "zeros"
            if variational_approx_params is None:
                variational_approx_params = {'n_variational_layers': 2}
            self.m_fun, self.S_fun = mean_field(self.K, self.effective_K, **variational_approx_params, proportion=proportion)

    @abstractmethod
    def forward(self, X):
        pass

    def encode(self, X):
        return self.forward(X)[:2]

    def decode(self, Z, O):
        X = torch.zeros_like(Z)
        batch_size = Z.size(0)

        for layer, K_l in enumerate(self.K):
            Z_l = Z[:, layer, :K_l]
            # If we are at the root, X_1 is sampled of P(exp(Z_1 + O))
            if layer == 0:
                exp_Z_l = exp(Z_l + O.view(batch_size, 1))
                for batch_index in range(batch_size):
                    X[batch_index, layer, :K_l] = Poisson(rate=exp_Z_l[batch_index]).sample()
            else:
                # If we're not at the root, based on Z_l, we sample each C(X_k^(l-1))
                node_parents = [node for node in self.tree.getNodesAtDepth(self.selected_layers[0] + layer - 1)]
                for k, node_parent in enumerate(node_parents):
                    X_parent = X[:, layer - 1, node_parent.layer_index]
                    children_index = [child.layer_index for child in node_parent.children]
                    if len(children_index) == 1:
                        X[:, layer, children_index[0]] = X_parent
                    else:
                        Z_l_child = Z_l[:, children_index]
                        Z_l_child_max = torch.max(Z_l_child, dim=1)[0].unsqueeze(-1).repeat(1, len(children_index))
                        probabilities = torch.softmax(Z_l_child - Z_l_child_max, dim=1)
                        for batch_index in range(batch_size):
                            try:
                                if X_parent[batch_index] > 0:
                                    # Torch multinomial does not support well high values, we use numpy instead
                                    X[batch_index, layer, children_index] = torch.tensor(np.random.multinomial(
                                        n=int(X_parent[batch_index]), pvals=probabilities[batch_index].detach().numpy()
                                    ), dtype=torch.float64)
                            except:
                                print('--------- ERROR ---------')
                                print('Batch index: i = ', batch_index)
                                print('Layer: l = ', layer)
                                print('Parent value: X_p =', int(X_parent[batch_index]))
                                print('Latents parent: Z_p + O = ',
                                      Z[batch_index, layer - 1, node_parent.layer_index].item() + O[batch_index].item())
                                print('Softmax: \n', probabilities[batch_index].item())
                                print('Latents: \n', Z_l_child[batch_index].item())
                                raise ValueError("Error in the multinomial sampling.")

        return X

    def offset_model_objective(self, X, O, offsets_variables):
        elbo = 0.
        batch_size = X.size(0)
        norm_factor = float(batch_size * len(self.K))
        if self.offset_method == "gaussian":
            offset_m, offset_log_s = offsets_variables
            # Squeezing enables the computation to be shape consistent by multiplication
            offset_log_var = self.offset_log_var.squeeze()
            elbo += - batch_size * offset_log_var / norm_factor
            elbo += offset_log_s.sum() / (2 * norm_factor)
            elbo += - ((self.offset_mean - offset_m) ** 2 + torch.exp(offset_log_s)).sum() / (
                        torch.exp(offset_log_var) * norm_factor)
            elbo += -(1 + np.log(2 * np.pi)) * batch_size / norm_factor
        elif self.offset_method == "gmm":
            offset_m, offset_log_s = offsets_variables
            elbo += 2 * torch.log(
                density_gaussian_mixture(
                    self.offset_mixture_weights(),
                    self.offset_mixture_means,
                    self.offset_mixture_log_vars,
                    O
                )
            ).mean() / norm_factor
            elbo += 0.5 * offset_log_s.sum() / norm_factor
            elbo += 0.5 * (1 + np.log(2 * np.pi)) * batch_size / norm_factor
        return elbo

    def objective(self, X, output):
        Z, O, m, log_S, mu, Omega, offsets_variables = output
        # Initializing the elbo count
        elbo = 0.
        # Initializing the normalization factors
        batch_size = X.size(0)
        norm_factor = float(batch_size * len(self.K))
        for layer, mask in enumerate(self.layer_masks):
            # Filter out the lonely children
            X_l = X[:, layer, mask]
            m_l, log_S_l = m[layer], log_S[layer]
            mu_l, Omega_l = mu[layer], Omega[layer]
            # If we are at the PLN layer, we should consider the offset
            if layer == self.pln_layer:
                mu_l = mu_l + O.view(-1, 1)
                m_l = m_l + O.view(-1, 1)

            # S_l is diagonal so the determinant is fast to compute
            # det_S_l = prod(diagonal(S_l, dim1=-2, dim2=-1), dim=1)
            log_det_S_l = sum(log_S_l, dim=1)
            S_l = exp(log_S_l)

            M = (mu_l - m_l).unsqueeze(-1)
            Sigma_hat = M @ M.mT + torch.diag_embed(S_l, dim1=-2, dim2=-1)

            log_det_Omega_l = torch.logdet(Omega_l)
            trace_SigmaOmega = batch_trace(Sigma_hat @ Omega_l)

            elbo += 0.5 * sum(log_det_Omega_l - trace_SigmaOmega + log_det_S_l) / norm_factor
            elbo += sum(X_l * m_l) / norm_factor
            if layer == 0:
                elbo += -sum(exp(m_l + S_l / 2)) / norm_factor
            else:
                node_parents = [node for node in self.tree.getNodesAtDepth(self.selected_layers[0] + layer - 1)]
                for node_parent in node_parents:
                    parent_index = node_parent.layer_index
                    children_index = [child.layer_index for child in node_parent.children]
                    # If there is only one child, we filter it out
                    if len(children_index) == 1:
                        continue
                    X_l_parent = X[:, layer - 1, parent_index]
                    Z_l_child = Z[:, layer, children_index]
                    elbo += -sum(X_l_parent * logsumexp(Z_l_child, dim=1)) / norm_factor
            elbo += -batch_size * self.effective_K[layer] / (2 * norm_factor)

        # The ELBO accounts for a log factorial term: sum log(X_L!) = sum log(gamma(X_L + 1))
        X_L = X[:, -1, :self.K[-1]]
        elbo += -torch.lgamma(X_L + 1).sum() / norm_factor

        # Compute the offset model if enabled
        elbo += self.offset_model_objective(X, O, offsets_variables)

        # The stochastic ELBO is not affected by the choice of the variational approximation as such, because we
        # estimate it using a single sample. However, the said sample is affected by the choice of the variational
        # approximation, and thus the ELBO is affected indirectly.
        return -elbo

    @abstractmethod
    def update_close_forms(self, X, output):
        """
        Optimize the PLN parameters using closed form expression
        """
        pass

    def posterior_sample_offsets(self, X):
        if self.offset_method in ["zeros", "logsum"]:
            return offsets(X, method=self.offset_method), None
        elif self.offset_method == "constant":
            O = self.offset_constant * torch.ones(X.size(0))
            return O, None
        elif self.offset_method == "gaussian":
            sum_log_X = torch.log(X[:, 0, :self.K[0]].sum(dim=1) + 1e-8).unsqueeze(-1)
            offset_m = self.offset_m_fun(sum_log_X)
            offset_log_s = self.offset_log_s_fun(sum_log_X)
            O = offset_m + torch.exp(0.5 * offset_log_s) * torch.randn_like(sum_log_X)
            return O, (offset_m, offset_log_s)
        elif self.offset_method == "gmm":
            # If we don't have labels at our disposal,
            # the posterior model is approximated by a single gaussian in the variational approximation
            sum_log_X = torch.log(X[:, 0, :self.K[0]].sum(dim=1) + 1e-8).unsqueeze(-1)
            # No random component is selected, we always use the same gaussian
            component = 0
            # Compute the parameters of the gaussian
            offset_m = self.offset_mixture_m_fun[component](sum_log_X)
            offset_log_s = self.offset_mixture_log_s_fun[component](sum_log_X)
            # Sample the offset using a gaussian with the computed parameters
            O = offset_m + torch.exp(0.5 * offset_log_s) * torch.randn_like(sum_log_X)
            return O, (offset_m, offset_log_s)
        return None

    def sample_offsets(self, batch_size):
        if self.offset_method == "zeros":
            return torch.zeros(batch_size)
        if self.offset_method == "constant":
            return self.offset_constant * torch.ones(batch_size)
        elif self.offset_method == "gaussian":
            return self.offset_mean + torch.exp(0.5 * self.offset_log_var) * torch.randn(batch_size)
        elif self.offset_method == "gmm":
            return sample_gaussian_mixture(self.offset_mixture_weights(), self.offset_mixture_means,
                                           self.offset_mixture_log_vars, batch_size)
        return None

    def above_layers_autofill(self, X):

        K_full = list(self.tree.getLayersWidth().values())[:self.selected_layers[1] + 1]
        X_abundance = torch.zeros(
            (X.size(0), len(K_full), max(self.K))
        )
        for i, X_i in enumerate(X):
            X_abundance_i = partial_abundance_matrix_to_tree(self.tree, X_i.numpy()).to_array()
            X_abundance[i, :, :] = torch.tensor(X_abundance_i)

        return X_abundance

    def theta_parameters(self):
        # Initialize a list to store the parameters
        theta_params = []
        # Iterate over each module in mu_fun and omega_fun
        for module in self.mu_fun:
            theta_params.extend(module.parameters())
        for module in self.omega_fun:
            theta_params.extend(module.parameters())
        # Return the parameters
        return theta_params

    def phi_parameters(self):
        # Initialize a list to store the parameters
        phi_params = []
        # Iterate over each module in m_fun and S_fun
        for module in self.m_fun:
            phi_params.extend(module.parameters())
        for module in self.S_fun:
            phi_params.extend(module.parameters())
        # Return the parameters
        return phi_params
