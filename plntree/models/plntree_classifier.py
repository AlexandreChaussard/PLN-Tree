import torch
import torch.nn as nn

from plntree.models.base import BaseModel
from plntree.models.plntree import PLNTree
from plntree.utils.model_utils import Preprocessing, offsets

from plntree.utils.classifiers import LSTMClassifier


class PLNTreeClassifier(PLNTree):

    def __init__(
            self,
            tree,
            n_classes,
            selected_layers=None,
            diag_smoothing_factor=0.1,
            diagonal_model=False,
            positive_fun="softplus",
            use_smart_init=True,
            variational_approx="mean_field",
            offset_method='zeros',
            n_variational_layers=1,
            n_latent_layers=1,
            classifier='lstm',
            normalize=True,
            standardize=False
    ):
        if offset_method == 'gmm':
            offset_method = f'gmm_{n_classes}'
        PLNTree.__init__(
            self,
            tree=tree,
            selected_layers=selected_layers,
            diag_smoothing_factor=diag_smoothing_factor,
            diagonal_model=diagonal_model,
            positive_fun=positive_fun,
            use_smart_init=use_smart_init,
            variational_approx=variational_approx,
            offset_method=offset_method,
            n_variational_layers=n_variational_layers,
            n_latent_layers=n_latent_layers,
        )
        BaseModel.__init__(self, True)
        self.n_classes = n_classes

        # Define the parameters of the classifier which is an LSTM
        self.preprocessing = Preprocessing(max(self.K), standardize=standardize, normalize=normalize)
        if classifier == 'lstm':
            self.classifier = LSTMClassifier(max(self.K), 32, 2, n_classes)
        else:
            self.classifier = classifier

    def pi(self, Z):
        Z = self.preprocessing(Z)
        return self.classifier(Z)

    def forward(self, X, Y):
        # Compute the label conditional outputs
        output = PLNTree.forward(self, X)
        Z = output[0]
        # Compute the probabilities of each class
        probas = self.pi(Z)
        # Compute the offsets
        O, offset_params = self.posterior_sample_offsets(X, Y)
        return Z, O, *output[2:-1], offset_params, probas

    def encode(self, X, Y):
        output = PLNTree.forward(self, X)
        Z = output[0]
        O, offset_params = self.posterior_sample_offsets(X, Y)
        return Z, O

    def objective(self, X, Y, output):
        output_generative = output[:-1]
        probas = output[-1]
        # Compute the generative part of the ELBO: L_{|Y, O} + L_{O|X}
        elbo_generative = PLNTree.objective(self, X, output_generative)
        # Compute the cross entropy term: L_{Y|Z}
        labels = Y.argmax(dim=1)
        elbo_classifier = nn.functional.cross_entropy(probas, labels, reduction='mean')
        return elbo_classifier + len(self.K) * elbo_generative

    def update_close_forms(self, X, Y, output):
        output_generative = output[:-1]
        PLNTree.update_close_forms(self, X, output_generative)

    def predict_proba(self, X, n_sampling=20):
        # Perform importance sampling to estimate the probabilities
        # For each sample i, draw N_sampling Z to compute E_Z[pi(Z)]
        probas = 0
        for j in range(n_sampling):
            Z, O = self.encode(X, None)
            probas += self.pi(Z)
        probas /= n_sampling
        return probas

    def predict(self, X, n_sampling=20):
        return self.predict_proba(X, n_sampling=n_sampling).argmax(dim=1)

    def posterior_sample_offsets(self, X, Y=None):
        if self.offset_method == "gmm":
            sum_log_X = torch.log(X[:, 0, :self.K[0]].sum(dim=1) + 1e-8).unsqueeze(-1)
            # Get the component associated to each sample
            component = Y.argmax(dim=1)
            # Compute the parameters of each gaussian associated to the drawn component
            offset_m = self.offset_mixture_m_fun[component](sum_log_X)
            offset_log_s = self.offset_mixture_log_s_fun[component](sum_log_X)
            # Sample the offset using a gaussian with the computed parameters
            O = offset_m + torch.exp(0.5 * offset_log_s) * torch.randn_like(sum_log_X)
            return O, (offset_m, offset_log_s)
        return PLNTree.posterior_sample_offsets(self, X)

    def sample(self, batch_size, offsets=None):
        X, Z, O = PLNTree.sample(self, batch_size, offsets)
        Y = torch.distributions.Categorical(self.pi(Z)).sample()
        return X, Y, Z, O