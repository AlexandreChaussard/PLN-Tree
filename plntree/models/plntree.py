import torch
import torch.nn as nn
from torch import transpose

from plntree.models.base_plntree import _PLNTree
from plntree.utils import seed_all
from plntree.utils.functions import clr_transform
from plntree.utils.model_utils import (PLNParameter, BoundLayer, PositiveDefiniteMatrix,
                                       Vect1OrthogonalProjectorHierarchical, progressive_NN, is_not_inf_not_nan)
from plntree.utils.variational_approximations import (VariationalApproximation,
                                                      amortized_backward_markov, backward_branch_markov,
                                                      weak_backward_markov, residual_backward_markov)


class PLNTree(nn.Module, _PLNTree):

    def __init__(
            self,
            tree,
            selected_layers=None,
            variance_bounds=(1e-8, 10.),
            mean_bounds=(-12., 12.),
            diag_smoothing_factor=0.1,
            diagonal_model=False,
            positive_fun="softplus",
            use_smart_init=True,
            variational_approx="amortized_backward",
            offset_method='zeros',
            n_latent_layers=1,
            variational_approx_params=None,
            identifiable=True,
            seed=None
    ):
        super(PLNTree, self).__init__()
        _PLNTree.__init__(self,
                          tree=tree, selected_layers=selected_layers, variance_bounds=variance_bounds,
                          mean_bounds=mean_bounds, diagonal_model=diagonal_model,
                          use_smart_init=use_smart_init, variational_approx=variational_approx,
                          offset_method=offset_method, variational_approx_params=variational_approx_params,
                          pln_layer=0, seed=seed
                          )
        self.diag_smoothing_factor = diag_smoothing_factor

        # mu_theta^l depends on Z_i^{l-1}
        # self.mu_1 = nn.Parameter(torch.zeros(self.effective_K[0]))
        self.mu_fun = nn.ModuleList()
        self.omega_fun = nn.ModuleList()

        for layer in range(len(self.effective_K)):
            if layer == 0:
                mu_module = PLNParameter(size=self.effective_K[layer], shift=1.)
                omega_module = PLNParameter(data=torch.eye(self.effective_K[layer]))
            else:
                K_l = self.effective_K[layer - 1]
                K_l_next = self.effective_K[layer]
                mu_module = nn.Sequential(
                    *progressive_NN(K_l, K_l_next, n_latent_layers),
                )
                projector = None
                if identifiable:
                    projector = Vect1OrthogonalProjectorHierarchical(
                        self.tree,
                        layer + self.selected_layers[0],
                        self.effective_K[layer]
                    )
                    mu_module.append(projector)
                else:
                    mu_module.append(BoundLayer(-100, 12, smoothing_factor=0.05))
                omega_module = nn.Sequential(
                    *progressive_NN(K_l, K_l_next * (K_l_next + 1) // 2, n_latent_layers),
                    PositiveDefiniteMatrix(min_diag=diag_smoothing_factor,
                                           diagonal=diagonal_model,
                                           positive_diag=positive_fun,
                                           projector=projector),
                )
            self.mu_fun.append(mu_module)
            self.omega_fun.append(omega_module)

        # We compute the maximum layer index
        self.L = len(self.effective_K) - 1

        # If the variational approximation is Markov, it depends on the X^{1:l} and Z^{l + 1}
        if self.variational_approx == VariationalApproximation.AMORTIZED_BACKWARD:
            if variational_approx_params is None:
                variational_approx_params = {
                    'embedder_type': 'GRU',
                    'embedding_size': 32,
                    'n_embedding_layers': 2,
                    'n_embedding_neurons': 32,
                    'n_after_layers': 1,
                    'preprocessing': ['log']
                }
            self.m_fun, self.S_fun, self.embedder = amortized_backward_markov(
                input_size=max(self.K),
                effective_K=self.effective_K,
                **variational_approx_params
            )
        elif self.variational_approx == VariationalApproximation.RESIDUAL_BACKWARD:
            if variational_approx_params is None:
                variational_approx_params = {
                    'embedder_type': 'GRU',
                    'embedding_size': 32,
                    'n_embedding_layers': 2,
                    'n_embedding_neurons': 32,
                    'n_after_layers': 1,
                    'preprocessing': ['log']
                }
            self.m_fun, self.S_fun, self.embedder = residual_backward_markov(
                input_size=max(self.K),
                K=self.K,
                effective_K=self.effective_K,
                **variational_approx_params
            )
        elif self.variational_approx == VariationalApproximation.WEAK_BACKWARD:
            if variational_approx_params is None:
                variational_approx_params = {
                    'n_layers': 2,
                    'preprocessing': ['log']
                }
            self.m_fun, self.S_fun = weak_backward_markov(
                K=self.K,
                effective_K=self.effective_K,
                **variational_approx_params
            )
        elif self.variational_approx == VariationalApproximation.BRANCH:
            self.m_fun, self.S_fun = backward_branch_markov(
                tree=self.tree,
                selected_layers=self.selected_layers,
                layer_masks=self.layer_masks,
            )

    def posterior_latent_sample(self, X, seed=None):
        seed_all(seed)
        Z, m, log_S = [], [], []
        batch_size = X.size(0)

        def reparametrization_trick(m, log_S_vec):
            # Reparametrization trick
            # We deal with S_l as if it was the variance
            std = torch.exp(0.5 * log_S_vec)
            eps = torch.randn_like(std)
            Z_ = m + eps * std
            return Z_

        # We start by generating Z, then we will compute the parameters of the posterior
        if self.variational_approx == VariationalApproximation.MEAN_FIELD:
            for layer, mask in enumerate(self.layer_masks):
                # Compute the parameters of Z ~ N(mu, Sigma²)
                X_l = X[:, layer, :self.K[layer]].type(torch.float64)
                m_fun, log_S_fun = self.m_fun[layer], self.S_fun[layer]
                m_l, log_S_l_vec = m_fun(X_l), log_S_fun(X_l)
                Z_l = reparametrization_trick(m_l, log_S_l_vec)
                # Embed Z_l in an identifiable form with the tree
                Z_l_embed = torch.zeros((batch_size, X.size(2)))
                Z_l_embed[:, mask] += Z_l

                m.append(m_l)
                log_S.append(log_S_l_vec)
                Z.append(Z_l_embed)
        elif 'BACKWARD' in self.variational_approx.name.upper():
            for index in range(len(self.layer_masks)):
                # Compute the parameters of Z ~ N(mu, Sigma²)
                layer = self.L - index
                if self.variational_approx == VariationalApproximation.AMORTIZED_BACKWARD or \
                        self.variational_approx == VariationalApproximation.RESIDUAL_BACKWARD:
                    if self.variational_approx == VariationalApproximation.AMORTIZED_BACKWARD:
                        X_1tol = X[:, :layer + 1, :].type(torch.float64)
                    else:
                        X_1tol = X[:, :layer + 1, :].type(torch.float64)
                    X_embed = self.embedder(X_1tol)
                    if self.variational_approx == VariationalApproximation.RESIDUAL_BACKWARD:
                        X_l_flat = X[:, layer, :self.K[layer]].type(torch.float64)
                        X_embed = torch.concat([X_l_flat, X_embed], dim=1)
                elif self.variational_approx == VariationalApproximation.WEAK_BACKWARD:
                    X_embed = X[:, layer, :self.K[layer]].type(torch.float64)
                else:
                    raise "Variational approximation not implemented"
                m_fun, S_fun = self.m_fun[layer], self.S_fun[layer]
                # The last layer is only getting X_embed (layer = 0 is actually layer = L, we're going backward)
                if layer == self.L:
                    m_l, log_S_l_vec = m_fun(X_embed), S_fun(X_embed)
                else:
                    # Z_l is computed in a backward fashion, so Z^{l + 1} is the previous "Z" element
                    Z_l_next = Z[-1][:, self.layer_masks[layer + 1]]
                    data_input = torch.cat([X_embed, Z_l_next], dim=-1)
                    m_l, log_S_l_vec = m_fun(data_input), S_fun(data_input)
                Z_l = reparametrization_trick(m_l, log_S_l_vec)
                # Embed Z_l in an identifiable form with the tree
                Z_l_embed = torch.zeros((batch_size, X.size(2)), dtype=torch.float64)
                Z_l_embed[:, self.layer_masks[layer]] = Z_l

                m.append(m_l)
                log_S.append(log_S_l_vec)
                Z.append(Z_l_embed)

            # We need to reverse the list of vectors in Z to get the right order since it was generated backward
            # Same for all the components related to q_phi
            Z = list(reversed(Z))
            m = list(reversed(m))
            log_S = list(reversed(log_S))
        elif self.variational_approx == VariationalApproximation.BRANCH:
            for index in range(len(self.layer_masks)):
                # Compute the parameters of Z ~ N(mu, Sigma²)
                layer = self.L - index
                m_fun, S_fun = self.m_fun[layer], self.S_fun[layer]
                # The last layer is only getting X^{1:L} (layer = 0 is actually layer = L, we're going backward)
                if layer == self.L:
                    X_L = X[:, -1, :].type(torch.float64)
                    m_l, log_S_l_vec = m_fun(X_L), S_fun(X_L)
                else:
                    # Z_l is computed in a backward fashion, so Z^{l + 1} is the previous "Z" element
                    Z_l_next = Z_l_embed
                    m_l, log_S_l_vec = m_fun(X, Z_l_next), S_fun(X, Z_l_next)
                Z_l = reparametrization_trick(m_l, log_S_l_vec)
                # Embed Z_l in an identifiable form with the tree
                Z_l_embed = torch.zeros((batch_size, X.size(2)))
                Z_l_embed[:, self.layer_masks[layer]] = Z_l

                m.append(m_l)
                log_S.append(log_S_l_vec)
                Z.append(Z_l_embed)

            # We need to reverse the list of vectors in Z to get the right order since it was generated backward
            # Same for all the components related to q_phi
            Z = list(reversed(Z))
            m = list(reversed(m))
            log_S = list(reversed(log_S))

        return torch.stack(Z, dim=1), m, log_S

    def forward(self, X, posterior_latent_sample=None, seed=None):
        mu, Omega = [], []
        batch_size = X.size(0)

        # Inject potential external variational approximation (useful for conditional models)
        if posterior_latent_sample is None:
            posterior_latent_sample = self.posterior_latent_sample

        Z, m, log_S = posterior_latent_sample(X, seed=seed)

        # We compute the parameters of the posterior after Z, as it may have been computed backward
        for layer, mask in enumerate(self.layer_masks):
            if layer == 0:
                mu_l = self.mu_l(layer, None, batch_size=batch_size)
                Omega_l = self.omega_l(layer, None, batch_size=batch_size)
            else:
                Z_l_prev = Z[:, layer - 1, self.layer_masks[layer - 1]]
                mu_l, Omega_l = self.mu_l(layer, Z_l_prev), self.omega_l(layer, Z_l_prev)

            mu.append(mu_l)
            Omega.append(Omega_l)

        # Compute the offsets
        O, offsets_variables = self.posterior_sample_offsets(X)

        return Z, O, m, log_S, mu, Omega, offsets_variables

    def update_close_forms(self, X, output):
        Z, _, m, log_S, _, _, _ = output
        # Close form optimization for the PLN layer
        latent_means = m[0]
        # mu_1 is given by the mean of the latents mean
        self.mu_fun[0].data = latent_means.mean(axis=0)
        # Omega is given by the inverse of the mean of the empirical covariances
        S = torch.exp(log_S[0])
        # stacking mu new values based on Z batch size (no other impact)
        M = (self.mu_fun[0](Z[:, 0]) - latent_means).unsqueeze(-1)
        Sigma_hat = M @ M.mT + torch.diag_embed(S, dim1=-2, dim2=-1)
        self.omega_fun[0].data = torch.inverse(Sigma_hat.mean(axis=0))

    def smart_init(self, dataloader):
        X = torch.cat([data[0][:, 0, :self.K[0]] for data in dataloader], dim=0)
        log_X_centered = torch.log(X + 1e-8) - torch.log(X + 1e-8).mean(axis=0)
        n_samples = X.size(0)
        Sigma_hat = log_X_centered.T @ log_X_centered / (n_samples - 1) + self.diag_smoothing_factor * torch.eye(
            self.K[0])
        self.omega_fun[0].data = torch.inverse(Sigma_hat)
        self.mu_fun[0].data = torch.log(X + 1e-8).mean(axis=0)
        return self

    def identify(self, Z):
        Z_proj = torch.zeros_like(Z)
        for layer, mask in enumerate(self.layer_masks):
            if layer == 0:
                Z_proj[:, layer, mask] = Z[:, layer, mask]
                continue
            projector = Vect1OrthogonalProjectorHierarchical(
                            self.tree,
                            layer + self.selected_layers[0],
                            self.effective_K[layer]
                        )
            Z_proj[:, layer, mask] = projector(Z[:, layer, mask])
        return Z_proj


    def sample(self, batch_size, offsets=None, seed=None):
        seed_all(seed)
        K_max = max(self.K)
        n_layers = len(self.K)
        if offsets is None:
            O = self.sample_offsets(batch_size=batch_size).unsqueeze(-1)
        else:
            O = offsets
        # We sample Z, then we will decode it into X
        Z = torch.zeros((batch_size, n_layers, K_max))
        for layer, mask in enumerate(self.layer_masks):
            if layer == 0:
                mu_l = self.mu_l(layer, None, batch_size=batch_size)
                # We add the constraint that omega_1 is a positive definite here
                Omega_l = self.omega_l(layer, None, batch_size=batch_size)
            else:
                Z_l_prev = Z[:, layer - 1, self.layer_masks[layer - 1]]
                mu_l, Omega_l = self.mu_l(layer, Z_l_prev), self.omega_l(layer, Z_l_prev)

            Omega_l_cholesky = torch.cholesky(Omega_l, upper=False)
            Omega_l_cholesky_T = transpose(Omega_l_cholesky, dim0=-2, dim1=-1)
            # If we want Z ~ N(mu, Omega^-1) and that we have Omega = LL.T
            # Then Z = mu + L.-T eps works
            # Hence we sample eps ~ N(0, 1)
            # Then we solve L.T eps_tild = eps so that eps_tild ~ N(0, Omega^-1)
            # Then Z = mu + eps_tild
            eps = torch.randn_like(mu_l).unsqueeze(-1)
            eps = torch.linalg.solve_triangular(Omega_l_cholesky_T, eps, upper=True).squeeze(-1)
            if is_not_inf_not_nan(eps):
                Z[:, layer, mask] = mu_l + eps
            else:
                mask = torch.isnan(eps) | torch.isinf(eps)
                row_mask = torch.any(mask, dim=1)
                valid_rows = ~row_mask
                filtered_eps = eps[valid_rows]
                filtered_mu = mu_l[valid_rows]
                Z_l_filtered = torch.add(filtered_eps, filtered_mu)
                new_size = min(Z_l_filtered.size(0), Z.size(0))
                Z = Z[:new_size, :, :]
                Z_l_filtered = Z_l_filtered[:new_size]
                Z[:, layer, mask] = Z_l_filtered
                print("WARNING: Cholesky matrix could not be inverted for all samples.")
                print("Generated batch size:", new_size)

        # We can not generate samples that have a seed parameter over exp(threshold_multinomial) numerically using Multinomial
        threshold_multinomial = 25
        accepted_samples_mask = torch.all(Z[:, 0, :self.K[0]] < threshold_multinomial, dim=1)
        if sum(accepted_samples_mask) != Z.size(0):
            print(f"WARNING: Some samples have a seed parameter over exp({threshold_multinomial}) and will be refused.")
            Z = Z[accepted_samples_mask]
            O = O[accepted_samples_mask]
            print("Generated batch size:", Z.size(0))

        # Simply unpack Z as X
        X = self.decode(Z, O, seed=seed)
        return X, Z, O

    def mu_l(self, layer, Z_l_prev, batch_size=1):
        if layer == 0:
            constant = torch.ones((batch_size, 1))
            return self.mu_fun[0](constant)
        else:
            return self.mu_fun[layer](Z_l_prev)

    def omega_l(self, layer, Z_l_prev, batch_size=1):
        if layer == 0:
            constant = torch.ones((batch_size, 1))
            return self.omega_fun[0](constant)
        else:
            return self.omega_fun[layer](Z_l_prev)

    def latent_tree_counts(self, Z, proportions=False):
        Z_post = torch.zeros_like(Z)
        for layer, mask in enumerate(self.layer_masks):
            if layer == 0:
                if proportions:
                    # If we want a proportion latent variable, we apply a softmax on the first layer
                    # which suggest the total count is 0 (Poisson under constrain of total count summing to 1)
                    Z_post[:, layer, mask] = torch.softmax(Z[:, layer, mask], dim=-1)
                else:
                    # If we want a count interpretation, we take the exponential of the latent variable
                    Z_post[:, layer, mask] = torch.exp(Z[:, layer, mask])
            else:
                # For each parent node, we allocate the latent variables to the children based on the children weights
                for parent in self.tree.getNodesAtDepth(layer + self.selected_layers[0] - 1):
                    Z_parent = Z_post[:, layer - 1, parent.layer_index].unsqueeze(-1)
                    children_index = [child.layer_index for child in parent.children]
                    if len(children_index) == 1:
                        Z_post[:, layer, children_index] = Z_parent
                    else:
                        weights = torch.softmax(Z[:, layer, children_index], dim=-1)
                        Z_post[:, layer, children_index] = Z_parent * weights
        return Z_post

    def latent_tree_allocation(self, Z):
        Z_post = torch.zeros_like(Z) + Z
        for layer, mask in enumerate(self.layer_masks):
            if layer == 0:
                continue
            else:
                # For each parent node, if they have an only child, we allocate the latent variables to the children
                for parent in self.tree.getNodesAtDepth(layer + self.selected_layers[0] - 1):
                    Z_parent = Z_post[:, layer - 1, parent.layer_index].unsqueeze(-1)
                    children_index = [child.layer_index for child in parent.children]
                    if len(children_index) == 1:
                        Z_post[:, layer, children_index] = Z_parent
        return Z_post

    def latent_clr(self, Z):
        V = self.latent_tree_counts(Z)
        for layer, mask in enumerate(self.layer_masks):
            V[:, layer, mask] = clr_transform(V[:, layer, mask])
        return V
