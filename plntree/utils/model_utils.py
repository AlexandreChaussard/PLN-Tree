import pickle
import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import time


def seed_all(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)

def _format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def fit(model, optimizer, data_train_loader, n_epoch, max_grad_norm=1., verbose=10, writer=None, writer_fun=None):
    losses = []
    model.smart_init(data_train_loader)
    smooth_avg = []
    timer = time.time()
    for epoch in range(n_epoch):
        train_loss = 0
        for batch_idx, data in enumerate(data_train_loader):
            if len(data) == 2:
                X, Y = data
            elif len(data) == 3:
                X, O, Y = data
            optimizer.zero_grad()

            if model.classifier_mode:
                loss = model.objective(X, Y, model.forward(X, Y))
            else:
                loss = model.objective(X, model.forward(X))
            loss.backward()

            train_loss += loss.item()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            # Further models specific steps
            if getattr(model, 'update_close_forms', None) is not None:
                if model.classifier_mode:
                    model.update_close_forms(X, Y, model.forward(X, Y))
                else:
                    model.update_close_forms(X, model.forward(X))
        avg_loss = train_loss / (batch_idx + 1)
        if writer is not None:
            writer.add_scalar('1. ELBO/Train', avg_loss, epoch)
            if writer_fun is not None:
                writer_fun(writer, model, epoch)
        if len(smooth_avg) == verbose:
            formatted_time = _format_time((time.time() - timer) * (n_epoch - epoch) / verbose)
            print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, np.mean(smooth_avg)) + '    | Remaining Time: ' + formatted_time)
            smooth_avg = [avg_loss]
            timer = time.time()
        else:
            smooth_avg += [avg_loss]
        losses.append(avg_loss)
    if len(smooth_avg) > 0:
        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, np.mean(smooth_avg)))
    return model, losses


def fit_alternate(model, optimizer_theta, optimizer_phi, data_train_loader, n_epoch, max_grad_norm_phi=1.,
                  max_grad_norm_theta=1., verbose=10,
                  writer=None, writer_fun=None):
    losses = []
    model.smart_init(data_train_loader)
    smooth_avg = []
    for epoch in range(n_epoch):
        train_loss = 0
        for batch_idx, (X, Y) in enumerate(data_train_loader):
            # Zero the gradients of both optimizers
            optimizer_theta.zero_grad()
            optimizer_phi.zero_grad()

            # Compute the loss
            if model.classifier_mode:
                loss = model.objective(X, Y, *model.forward(X, Y))
            else:
                loss = model.objective(X, *model.forward(X))

            # Backpropagate the loss and update the parameters of the first group (phi)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.phi_parameters(), max_grad_norm_phi)
            optimizer_phi.step()

            # Zero the gradients of both optimizers again
            optimizer_theta.zero_grad()
            optimizer_phi.zero_grad()

            # Compute the loss again
            loss = model.objective(X, *model.forward(X))

            # Backpropagate the loss and update the parameters of the second group (theta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.theta_parameters(), max_grad_norm_theta)
            torch.nn.utils.clip_grad_norm_(model.phi_parameters(), max_grad_norm_phi)
            optimizer_theta.step()

            # Further models specific steps
            if getattr(model, 'update_close_forms', None) is not None:
                if model.classifier_mode:
                    model.update_close_forms(X, *model.forward(X, Y))
                else:
                    model.update_close_forms(X, *model.forward(X))

            train_loss += loss.item()

        avg_loss = train_loss / (batch_idx + 1)
        if writer is not None:
            writer.add_scalar('1. ELBO/Train', avg_loss, epoch)
            if writer_fun is not None:
                writer_fun(writer, model, epoch)
        if len(smooth_avg) == verbose:
            print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, np.mean(smooth_avg)))
            smooth_avg = [avg_loss]
        else:
            smooth_avg += [avg_loss]
        losses.append(avg_loss)
    if len(smooth_avg) > 0:
        print('[*] Epoch: {} Average loss: {:.4f}'.format(epoch, np.mean(smooth_avg)))
    return model, losses


def pickle_model(modelClass, params, optimizer, optim_params, dataloader, n_epoch, overwrite=False):
    fmodel = f"{modelClass.__name__}-{n_epoch}-{[f'{k}:{str(v)}' for k, v in params.items()]}.pkl" \
        .replace("'", '').replace(":", "=")
    if os.path.exists(fmodel) and not overwrite:
        model, losses = pickle.load(open(fmodel, 'rb'))
    else:
        model = modelClass(**params)
        model, losses = fit(model, optimizer(model.parameters(), **optim_params), dataloader, n_epoch=n_epoch)
        pickle.dump([model, losses], open(fmodel, 'wb'))

    return model, losses


def save_load(object, fileName, overwrite=False):
    fileName += '.pkl'
    if os.path.exists(fileName) and not overwrite:
        print('Loading object from file. Overwrite is set to false.')
        return pickle.load(open(fileName, 'rb'))
    else:
        if object is None:
            print('Object is None, not saving.')
            return
        directory = os.path.dirname(fileName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(object, open(fileName, 'wb'))
        print('Object saved successfully.')

def linear_builder(sizes, activation=nn.ReLU, output_activation=None):
    return nn.Sequential(
        *[nn.Sequential(
            nn.Linear(sizes[i], sizes[i + 1]),
            activation()
        ) for i in range(len(sizes) - 2)] + [nn.Linear(sizes[-2], sizes[-1])] + ([output_activation] if output_activation is not None else [])
    )

def batch_layers_to_tensors(Z_list, K, fill_value=0.):
    K_max = max(K)
    batch_size = len(Z_list[0])
    Z = torch.full((batch_size, len(K), K_max), fill_value)

    for layer, K_l in enumerate(K):
        Z[:, layer, :K_l] = Z_list[layer]

    return Z


def batch_symmetric_from_vector(vector, size):
    i, j = torch.triu_indices(size, size)
    A = torch.zeros((vector.size(0), size, size))
    A[:, i, j] = vector
    torch.transpose(A, dim1=-2, dim0=-1)[:, i, j] = vector
    return A


def backpropagate_gaussian_sample(mu, log_var):
    # this function samples a Gaussian distribution,
    # with average (mu) and standard deviation specified (using log_var)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)  # return z sample


def sample_multivariate_gaussian(mu, triangular, upper):
    # If we want Z ~ N(mu, Omega^-1) and that we have Omega = LL.T
    # Then Z = mu + L.-T eps works
    # Hence we sample eps ~ N(0, 1)
    # Then we solve L.T eps_tild = eps so that eps_tild ~ N(0, Omega^-1)
    # Then Z = mu + eps_tild
    batch_size = mu.size(0)
    eps = MultivariateNormal(
        loc=torch.zeros_like(mu),
        covariance_matrix=batch_identity(batch_size, mu.size(1))
    ).sample().unsqueeze(-1)
    check_not_inf_not_nan(eps)
    eps = torch.linalg.solve_triangular(triangular, eps, upper=upper).squeeze(-1)
    return eps + mu


def batch_trace(batch):
    return batch.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def batch_lower_triangular_to_matrix(L_vec, batch, size):
    # Tril to get the lower diagonal part
    i, j = torch.tril_indices(size, size)
    L = torch.zeros((batch, size, size))
    L[:, i, j] = L_vec
    return L


def batch_triangular_inverse(L, eps=0.):
    return torch.linalg.solve_triangular(
        L + eps * batch_identity(L.size(0), L.size(1)), batch_identity(L.size(0), L.size(1)),
        upper=False
    )


def lower_triangular_to_vec(M):
    K_1, K_2 = M.size(-2), M.size(-1)
    k, m = torch.tril_indices(K_1, K_2)
    return M[k, m]


def batch_identity(batch_size, size):
    return torch.eye(size).unsqueeze(0).expand(batch_size, -1, -1)


def flatten_taxaabundance(X, layer, K):
    size = sum(K[:layer + 1])
    X_flat = torch.zeros((X.size(0), size))
    K_cum_prev = 0
    for l, K_l in enumerate(K[:layer + 1]):
        K_cum = K_cum_prev + K_l
        X_flat[:, K_cum_prev:K_cum] = X[:, l, :K_l]
        K_cum_prev = K_cum
    return X_flat


def offsets(X, method):
    assert (method in ['logsum', 'zeros'])
    if method == 'logsum':
        return torch.tensor(torch.log(X.sum(axis=2)[:, 0]))
    else:
        return torch.zeros(X.size(0))


def modify_weights(module_list, factor=1., shift=0., at_layer=None):
    for seq in module_list:
        for layer in seq:
            if at_layer is not None and layer != at_layer:
                continue
            if isinstance(layer, nn.Linear):
                layer.weight.data *= factor
                layer.weight.data += shift


def softmax_taxonomy(Z, taxonomy):
    Z_softmax = torch.zeros_like(Z)
    for node in taxonomy.nodes:
        if node.hasChildren():
            children_layer = node.depth + 1
            children_index = [child.layer_index for child in node.children]
            Z_children = Z[:, children_layer, children_index]
            Z_softmax[:, children_layer, children_index] = torch.softmax(Z_children, dim=1).unsqueeze(-1)
    return Z_softmax


def softmax_taxonomy_layer(Z_l, layer, taxonomy):
    if layer == 0:
        return Z_l
    Z_softmax = torch.zeros_like(Z_l)
    for node in taxonomy.getNodesAtDepth(layer - 1):
        if node.hasChildren():
            children_index = [child.layer_index for child in node.children]
            Z_children = Z_l[:, children_index]
            Z_softmax[:, children_index] = torch.softmax(Z_children, dim=1)
    return Z_softmax


def positive_output(input, function=None, function_args=()):
    if function == "softplus":
        return nn.Softplus(*function_args)(input)
    if function == "square":
        return torch.pow(input, 2)
    if function == "abs":
        return torch.abs(input)
    if function == "smooth_abs":
        return torch.sqrt(torch.pow(input, 2) + 1e-8)
    else:
        return None


class LogTransform(nn.Module):
    def __init__(self, shift=1e-16):
        super(LogTransform, self).__init__()
        self.shift = shift

    def forward(self, X):
        return torch.log(X + self.shift)


class Preprocessing(nn.Module):

    def __init__(self, n_features, log=False, standardize=False, normalize=False, proportion=False):
        super(Preprocessing, self).__init__()
        self.n_features = n_features
        self.log_transform = LogTransform() if log else None
        self.standardize = nn.BatchNorm1d(n_features, affine=False) if standardize else None
        self.normalize = normalize
        self.proportion = proportion
        # TODO: Other preprocessing specific to count data are possible like
        #  log-ratio transformation, CLR, ALR, ILR, etc.

    def forward(self, X):
        x = X.clone()
        if self.proportion:
            x = x / (torch.sum(x, dim=0, keepdim=True) + 1e-15)
        if self.log_transform is not None and not self.proportion:
            x = self.log_transform(x)
        if self.standardize is not None:
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-15)
        if self.normalize and not self.proportion:
            x_max = X.max()
            x_min = X.min()
            x = (x - x_min) / (x_max - x_min)
        return x


class PartialPreprocessing(nn.Module):
    def __init__(self, n_features, log=False, standardize=False, normalize=False, proportion=False):
        super(PartialPreprocessing, self).__init__()
        self.n_features = n_features
        self.preprocessing = Preprocessing(n_features, log=log, standardize=standardize, normalize=normalize, proportion=proportion)

    def forward(self, input):
        X = input[:, :self.n_features]
        return torch.concat((self.preprocessing(X), input[:, self.n_features:]), dim=-1)


class ConstantSum(nn.Module):
    def __init__(self, constant=1.):
        super(ConstantSum, self).__init__()
        self.constant = constant

    def forward(self, input):
        return input / input.sum(-1, keepdim=True) - 1 + self.constant


class BoundLayer(nn.Module):
    def __init__(self, min_value, max_value, smoothing_factor=0.2):
        super(BoundLayer, self).__init__()
        self.min_value = min(min_value, max_value)
        self.max_value = max(max_value, min_value)
        self.smoothing_factor = smoothing_factor

    def forward(self, x):
        return (self.min_value + (self.max_value - self.min_value)
                * torch.sigmoid(self.smoothing_factor * (x - (self.min_value + self.max_value) / 2)))


class Vect1OrthogonalProjector(nn.Module):
    def __init__(self):
        super(Vect1OrthogonalProjector, self).__init__()

    def forward(self, X):
        # If it's a vector, project it on Vect(1_d)^orthogonal
        if len(X.shape) == 2:
            return X - X.mean(dim=-1, keepdim=True)
        # If it's a matrix, project it on Vect(1_{dxd})^orthogonal
        elif len(X.shape) == 3:
            d = X.size(-1)
            P = torch.eye(d) - torch.ones(d, d) / d
            return P @ X @ P
        return None


class Vect1OrthogonalProjectorHierarchical(nn.Module):

    def __init__(self, tree, layer, K_eff):
        super(Vect1OrthogonalProjectorHierarchical, self).__init__()
        self.tree = tree
        # Create the tensors of projection of shape [L-1, K_l, K_l]
        self.P = self.projector(layer, K_eff)

    def projector(self, layer, K_eff):
        P = torch.eye(K_eff)
        mapping_index = 0
        for parent in self.tree.getNodesAtDepth(layer - 1):
            d = len(parent.children)
            # If it's a lonely child, then it's not accounted for in the modelisation
            if d == 1:
                continue
            Q = torch.eye(d) - torch.ones(d, d) / d
            span = torch.arange(mapping_index, mapping_index + d)
            P[span, span.unsqueeze(1)] = Q.clone()
            mapping_index += d
        return P

    def forward(self, X):
        P = self.P.expand(X.size(0), -1, -1).to(dtype=X.dtype)
        # If it's a vector, project it on Vect(1_d)^orthogonal
        if len(X.shape) == 2:
            return (P @ X.unsqueeze(-1)).squeeze(-1)
        # If it's a matrix, project it on Vect(1_{dxd})^orthogonal
        elif len(X.shape) == 3:
            return P @ X @ P
        return None


class PositiveDefiniteMatrix(nn.Module):

    def __init__(self, min_diag=1e-8, diagonal=False, positive_diag="abs", projector=None):
        super(PositiveDefiniteMatrix, self).__init__()
        self.min_diag = min_diag
        self.positive_diag = positive_diag
        self.diagonal = diagonal
        self.projector = projector

    def forward(self, L_vec):
        cholesky = Cholesky(min_diag=0, diagonal=self.diagonal, positive_diag=self.positive_diag)
        L = cholesky(L_vec)
        if self.projector is not None:
            # The identifiability is given on Sigma as P Sigma P, so Omega = (P Sigma P)^-1
            # Hence we act as if we had built Sigma so far, and will simply invert the positive definite matrix
            L = self.projector.P.to(dtype=L.dtype) @ L
        Omega = L @ L.mT
        # Regularize the diagonal to ensure invertibility
        i = torch.arange(Omega.size(-1))
        Omega[:, i, i] = Omega[:, i, i] + self.min_diag
        if self.projector is not None:
            try:
                Omega = torch.inverse(Omega)
            except:
                print("Omega could not be inverted, here is the smallest eigenvalue:", torch.linalg.eigvalsh(Omega).min())
                print("Minimum threshold is: ", self.min_diag)
                print("The smallest determinant is: ", torch.det(Omega).min())
                L_ = cholesky(L_vec)
                batch_size = L_.size(0)
                PL_ = self.projector.P.to(dtype=L_.dtype).unsqueeze(0).expand(batch_size, -1, -1) @ L_
                Omega_ = PL_ @ PL_.mT
                print('Cholesky without projection minimum eigen value is: ', torch.diagonal(L_, dim1=-1).min())
                print("Projected Cholesky product minimum eigen value is: ", torch.linalg.eigvalsh(Omega).min())
                print("Symmetry check: \n", (Omega_ - Omega_.mT).sum())
                print("Maximum value in abs(L): ", L_.abs().max())
                print("Projected Cholesky \n", PL_)
                raise ValueError("Omega is singular.")
        return Omega


class Cholesky(nn.Module):

    def __init__(self, min_diag=1e-8, diagonal=False, positive_diag="abs"):
        super(Cholesky, self).__init__()
        self.min_diag = min_diag
        self.diagonal = diagonal
        self.positive_diag = positive_diag

    def forward(self, L_vec):
        size = -0.5 + (1 + 8 * L_vec.size(1)) ** .5 / 2
        assert (abs(size - int(size)) < 10e-8)
        size = int(size)
        batch = L_vec.size(0)
        # Tril to get the lower diagonal part and the diagonal
        i, j = torch.tril_indices(size, size)
        L = torch.zeros((batch, size, size), dtype=L_vec.dtype)
        L[:, i, j] = L_vec
        L_below = torch.tril(L, diagonal=-1)
        L_diag = torch.diagonal(L, dim1=-2, dim2=-1)
        # Make the diagonal positive to ensure a unique decomposition
        L_diag = positive_output(L_diag, self.positive_diag)
        # Embed the diagonal in a matrix
        L_diag = torch.diag_embed(L_diag, dim1=-2, dim2=-1)
        # Regularize the diagonal to ensure inversibility
        L_diag = L_diag + self.min_diag * batch_identity(batch, size)
        if self.diagonal:
            return L_diag
        L = L_diag + L_below
        # Triu to get the upper diagonal part out of the diagonal and set it to 0
        u, v = torch.triu_indices(size, size, offset=1)
        L[:, u, v] = 0.
        return L


class PLNParameter(nn.Module):
    def __init__(self, size=None, data=None, shift=0.):
        super(PLNParameter, self).__init__()
        if data is None:
            data = torch.randn(size) + shift
        self.data = data

    def forward(self, X):
        batch_size = X.size(0)
        repeat_dim = [1] * len(self.data.shape)
        return self.data.unsqueeze(0).repeat(batch_size, *repeat_dim)


class SimplexParameter(nn.Module):
    def __init__(self, size, seed=None):
        super(SimplexParameter, self).__init__()
        seed_all(seed)
        self.param = nn.Parameter(torch.randn(size))

    def forward(self):
        return torch.softmax(self.param, dim=0)


def sample_gaussian_mixture(pi, mu, log_var, n_samples, n_features=1):
    # Sample from a Gaussian mixture
    # pi: (n_components, )
    # mu: (n_components, n_features)
    # sigma: (n_components, n_features, n_features)
    n_components = pi.size(0)
    # Draw a component for each sample
    component = torch.multinomial(pi, n_samples, replacement=True)
    # Draw each sample from the corresponding component
    samples = torch.zeros((n_samples, n_features))
    for i in range(n_samples):
        samples[i] = mu[component[i]] + torch.exp(0.5 * log_var[component[i]]) * torch.randn(size=n_features)
    return samples


def density_gaussian_mixture(pi, mu, log_var, samples):
    # Compute the density of a Gaussian mixture
    # pi: (n_components, )
    # mu: (n_components, n_features)
    # sigma: (n_components, n_features, n_features)
    n_components = pi.size(0)
    n_samples = samples.size(0)
    density = torch.zeros(n_samples)
    for i in range(n_samples):
        for k in range(n_components):
            density[i] += pi[k] * torch.exp(
                MultivariateNormal(loc=mu[k], covariance_matrix=torch.exp(0.5 * log_var[k])).log_prob(samples[i]))
    return density


def progressive_NN(start_dim, end_dim, n_layers):
    module = []
    prev_dim = start_dim
    mid_dim = start_dim
    for k in range(n_layers):
        module.append(nn.Linear(prev_dim, mid_dim))
        module.append(nn.ReLU())
        prev_dim = mid_dim
    module.append(nn.Linear(prev_dim, end_dim))
    return module


def check_not_inf_not_nan(tensor, additionalInfo=""):
    value = not torch.isinf(tensor).any().item() and not torch.isnan(tensor).any().item()
    if not value:
        print("Assertion error, tensor matched inf or nan values: \n", tensor)
        print("Additional info: \n", additionalInfo)
    assert value


def is_not_inf_not_nan(tensor):
    return not torch.isinf(tensor).any().item() and not torch.isnan(tensor).any().item()
