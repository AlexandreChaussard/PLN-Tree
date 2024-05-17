import torch
import torch.nn as nn
import numpy as np
import hamiltorch

from plntree.models.plntree import PLNTree
from plntree.models.plntree_classifier import PLNTreeClassifier
from plntree.utils import seed_all


class PLNTreeConditional(PLNTreeClassifier):
    def __init__(
            self,
            tree,
            n_classes,
            selected_layers=None,
            identifiable=True,
            diag_smoothing_factor=0.1,
            diagonal_model=False,
            positive_fun="softplus",
            use_smart_init=True,
            variational_approx="mean_field",
            variational_approx_params=None,
            offset_method='gmm',
            n_latent_layers=1,
            classifier=None,
            seed=None,
    ):
        if offset_method == 'gmm':
            offset_method = f'gmm_{n_classes}'
        PLNTreeClassifier.__init__(
            self,
            tree=tree,
            n_classes=n_classes,
            selected_layers=selected_layers,
            diag_smoothing_factor=diag_smoothing_factor,
            diagonal_model=diagonal_model,
            positive_fun=positive_fun,
            use_smart_init=use_smart_init,
            variational_approx=variational_approx,
            variational_approx_params=variational_approx_params,
            offset_method=offset_method,
            identifiable=identifiable,
            n_latent_layers=n_latent_layers,
            classifier=classifier,
            seed=seed,
        )
        seeds = [None] * n_classes
        if seed is not None:
            seed_all(seed)
            seeds = np.random.randint(0, 2 ** 32, n_classes)
        # We need to define a variational approximation for each class (conditional variational approximation)
        self.conditional_forward_list = nn.ModuleList([
            PLNTree(
                tree=tree,
                selected_layers=selected_layers,
                diag_smoothing_factor=diag_smoothing_factor,
                diagonal_model=diagonal_model,
                positive_fun=positive_fun,
                use_smart_init=use_smart_init,
                variational_approx=variational_approx,
                offset_method=offset_method,
                n_latent_layers=n_latent_layers,
                variational_approx_params=variational_approx_params,
                identifiable=identifiable,
                seed=seeds[_]
            ) for _ in range(n_classes)
        ])

    def conditional_forward(self, X, Y, seed=None):
        # Find the index of the label
        seed_all(seed)
        labels = Y.argmax(dim=1)
        outputs = []
        for unique_label in torch.unique(labels):
            X_label = X[labels == unique_label]
            output = PLNTree.forward(
                self, X_label,
                posterior_latent_sample=self.conditional_forward_list[unique_label].posterior_latent_sample
            )
            outputs.append(output)
        return outputs

    def forward(self, X, Y, seed=None):
        seed_all(seed)
        # Compute the label conditional outputs
        outputs = self.conditional_forward(X, Y)
        outputs_cond = []
        labels = Y.argmax(dim=1)
        for c, unique_label in enumerate(torch.unique(labels)):
            output = outputs[c]
            Z = output[0]
            # Compute the probabilities
            probas = self.pi(Z)
            # Compute the offsets
            O, offset_params = self.posterior_sample_offsets(X[labels == unique_label], Y[labels == unique_label])
            outputs_cond.append((Z, O, *output[2:-1], offset_params, probas))
        return outputs_cond

    def objective(self, X, Y, outputs_cond):
        elbo = 0
        labels = Y.argmax(dim=1)
        for c, unique_label in enumerate(torch.unique(labels)):
            output = outputs_cond[c]
            output_generative = output[:-1]
            probas = output[-1]
            # Compute the generative part of the ELBO: L_{|Y, O} + L_{O|X}
            elbo_generative = PLNTree.objective(self, X[labels == unique_label], output_generative)
            # Compute the cross entropy term: L_{Y|Z}
            elbo_classifier = nn.functional.cross_entropy(probas, labels[labels == unique_label], reduction='mean')
            elbo += elbo_classifier + elbo_generative
        return elbo

    def update_close_forms(self, X, Y, outputs_cond):
        Z = None
        m_list = []
        log_S_list = []
        for output in outputs_cond:
            if Z is None:
                Z = output[0]
                m_list = output[2]
                log_S_list = output[3]
            else:
                Z = torch.cat((Z, output[0]), dim=0)
                for layer in range(len(self.K)):
                    m_list[layer] = torch.cat((m_list[layer], output[2][layer]), dim=0)
                    log_S_list[layer] = torch.cat((log_S_list[layer], output[3][layer]), dim=0)
        output_generative = (Z, None, m_list, log_S_list, None, None, None)
        PLNTree.update_close_forms(self, X, output_generative)

    def encode(self, X, Y, seed=None):
        seed_all(seed)
        Y_onehot = torch.nn.functional.one_hot(Y.to(torch.int64), self.n_classes).type(torch.float64)
        outputs_cond = self.conditional_forward(X, Y_onehot)
        labels = Y_onehot.argmax(dim=1)

        Z = torch.zeros_like(X)
        O = torch.zeros(X.size(0))
        for c, unique_label in enumerate(torch.unique(labels)):
            Z[labels == unique_label] = outputs_cond[c][0]
            O[labels == unique_label] = self.posterior_sample_offsets(X[labels == unique_label], Y_onehot[labels == unique_label])[0]
        return Z, O

    def gibbs_predict_proba(self, X, n_sampling=20, n_iter_gibbs=20, seed=None):
        seed_all(seed)
        # For each sample i, draw N_sampling Z from p(Z|X) to compute E[pi(Z) | X]
        # Perform Gibbs sampling to sample under p(Z|X)
        probas = 0
        # Importance sampling loop
        for j in range(n_sampling):
            # Gibbs sampler
            O = self.posterior_sample_offsets(X, None)[0]
            Z = self.sample(X.size(0), offsets=O)[2]
            for k in range(n_iter_gibbs):
                Y_labels = torch.distributions.Categorical(self.pi(Z)).sample()
                Z, O = self.encode(X, Y_labels)
            probas += self.pi(Z)
        probas /= n_sampling
        return probas

    def hmc_log_prob_target(self, Z, X):
        """
        Compute the log probability of the target distribution in the HMC for the prediction
        Normally, this would be log p(Z|X), but since we are only interested in its gradient,
        we can target the joint law log p(Z, X) instead, easily computed with the decomposition
        log p(Z, X) = log p(Z) + log p(X|Z)
        """
        # Reformat Z to match the shape of the model
        Z = Z.view(*X.size())
        # Compute the PLN layer's distribution
        Z_1 = Z[:, 0, self.layer_masks[0]]
        X_1 = X[:, 0, self.layer_masks[0]]
        eval = torch.distributions.Poisson(torch.exp(Z_1)).log_prob(X_1).sum()
        # Compute the multinomial propagations distribution
        for layer in range(0, len(self.L)):
            for node in self.tree.getNodesAtDepth(layer):
                children_index = [child.layer_index for child in node.children]
                Z_child = Z[:, layer+1, children_index]
                X_child = X[:, layer+1, children_index]
                X_parent = X[:, layer, node.layer_index]
                eval += torch.distributions.Multinomial(
                    total_count=X_parent,
                    probs=torch.nn.functional.softmax(Z_child, dim=1),
                ).log_prob(X_child).sum()
        # Compute the latents distribution
        # Starting with the first layer which is Gaussian
        eval += torch.distributions.MultivariateNormal(
            loc=self.mu_fun[0],
            precision_matrix=self.omega_fun[0],
        ).log_prob(Z_1).sum()
        # The propagation is Gaussian Markov
        for layer in range(0, len(self.L)-1):
            Z_prev = Z[:, layer, self.layer_masks[layer]]
            Z_cur = Z[:, layer+1, self.layer_masks[layer+1]]
            eval += torch.distributions.MultivariateNormal(
                loc=self.mu_fun[layer+1](Z_prev),
                precision_matrix=self.omega_fun[layer+1](Z_prev),
            ).log_prob(Z_cur).sum()
        return eval

    def predict_proba(self, X, n_sampling=20, hmc_step=0.3, hmc_n_steps=5, offsets=None, seed=None):
        seed_all(seed)
        Z_init = self.sample(X.size(0), offsets=offsets)[2]
        Z_hmc = hamiltorch.sample(
            log_prob_func=lambda Z: self.hmc_log_prob_target(Z, X),
            params_init=Z_init,
            num_samples=n_sampling,
            step_size=hmc_step,
            num_steps_per_sample=hmc_n_steps
        )
        Z_hmc = Z_hmc.view(n_sampling, *X.size())
        probas = 0
        for Z_proposal in Z_hmc:
            probas += self.pi(Z_proposal)
        probas /= n_sampling
        return probas

    def predict(self, X, n_sampling=20, n_iter_gibbs=20, seed=None):
        return self.predict_proba(X, n_sampling=n_sampling, n_iter_gibbs=n_iter_gibbs, seed=seed).argmax(dim=1)

    def sample(self, batch_size, offsets=None, seed=None):
        seed_all(seed)
        X, Z, O = PLNTree.sample(self, batch_size, offsets)
        Y = torch.distributions.Categorical(self.pi(Z)).sample()
        return X, Y, Z, O

    def gibbs_conditional_sample(self, label, batch_size, offsets=None, n_iter_gibbs=20, X_init=None, seed=None):
        seed_all(seed)
        batch_size = min(batch_size, n_iter_gibbs)
        labels = torch.tensor([label] * batch_size).type(torch.int64)
        Y = torch.nn.functional.one_hot(
            labels, self.n_classes
        ).type(torch.float64)
        if X_init is None:
            X_cur = self.sample(batch_size, offsets)[0]
        else:
            X_cur = X_init
        for _ in range(1, n_iter_gibbs):
            Z_cur, O_cur = self.encode(X_cur, labels)
            if offsets is not None:
                O_cur = offsets
            X_cur = self.decode(Z_cur, O_cur)
        return X_cur, Y, Z_cur, O_cur

    def conditional_sample(self, label, batch_size, offsets=None, seed=None):
        seed_all(seed)
        Y = torch.nn.functional.one_hot(
            torch.tensor([label] * batch_size).type(torch.int64), self.n_classes
        ).type(torch.float64)
        X, Z, O = [], [], []
        while len(X) < batch_size:
            X_, Y_, Z_, O_ = self.sample(batch_size, offsets)
            for y in Y_:
                if y == label:
                    X.append(X_)
                    Z.append(Z_)
                    O.append(O_)
                if len(X) == batch_size:
                    break
        return torch.cat(X, dim=0), Y, torch.cat(Z, dim=0), torch.cat(O, dim=0)
