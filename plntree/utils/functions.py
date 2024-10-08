import numpy as np
import torch
import random


def seed_all(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(int(seed))


def remove_outliers_iqr(df, column_name, threshold=1.5):
    """
    Remove outliers from a DataFrame using the IQR method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column containing the data.
    - threshold (float): IQR multiplier to determine the range for outliers.

    Returns:
    - pd.DataFrame: DataFrame without outliers.
    """
    # Calculate Q1 and Q3
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Filter out outliers
    df_no_outliers = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df_no_outliers


def log_pdf_multivariate_normal(mu, omega, Z):
    # eval += torch.distributions.MultivariateNormal(
    #    loc=m_cur,
    #    precision_matrix=omega_cur,
    # ).log_prob(Z_cur).sum()
    n = Z.shape[0]
    d = mu.shape[-1]
    eval = -n * d * np.log(2 * np.pi) / 2
    eval += torch.logdet(omega).sum()
    eval += -1 / 2 * ((Z - mu).unsqueeze(-1).mT @ omega @ (Z - mu).unsqueeze(-1)).squeeze().sum()
    assert not torch.isnan(eval).any()
    assert not torch.isinf(eval).any()
    return eval


def log_pdf_cond_pln(X, Z_cond):
    # eval = torch.distributions.Poisson(torch.exp(Z_1)).log_prob(X_1).sum()
    eval = (Z_cond * X - torch.exp(Z_cond) - torch.lgamma(X + 1)).sum()
    assert not torch.isnan(eval).any()
    assert not torch.isinf(eval).any()
    return eval


def log_pdf_multinomial_plntree(X_parent, X_child, Z_child):
    # for i in range(X.shape[0]):
    #    eval += torch.distributions.Multinomial(
    #        total_count=int(X_parent[i]),
    #        probs=torch.nn.functional.softmax(Z_child[i] - Z_child[i].max()),
    #    ).log_prob(X_child[i].to(torch.int64)).sum()
    eval = (torch.lgamma(X_parent + 1) - torch.lgamma(X_child + 1).sum(-1)).sum()
    eval += (Z_child * X_child).sum()
    eval += (-X_parent * torch.logsumexp(torch.exp(Z_child), dim=-1)).sum()
    assert not torch.isnan(eval).any()
    assert not torch.isinf(eval).any()
    return eval

def log_pdf_cond_plntree(plntree, X, Z_cond, O_cond):
    # Compute the PLN layer's distribution
    Z_1 = Z_cond[:, 0, plntree.layer_masks[0]] + O_cond
    X_1 = X[:, 0, plntree.layer_masks[0]]
    eval = log_pdf_cond_pln(X_1, Z_1)
    # Compute the multinomial propagations distribution
    for layer in range(0, len(plntree.K) - 1):
        for node in plntree.tree.getNodesAtDepth(layer + plntree.selected_layers[0]):
            children_index = [child.layer_index for child in node.children]
            Z_child = Z_cond[:, layer + 1, children_index] + O_cond
            X_child = X[:, layer + 1, children_index]
            X_parent = X[:, layer, node.layer_index]
            eval += log_pdf_multinomial_plntree(X_parent, X_child, Z_child)
    return eval

def log_pdf_plntree(plntree, X, Z, O):
    # Compute the observed counts log pdf
    eval = log_pdf_cond_plntree(plntree, X, Z, O)
    # Compute the latents Markov Gaussian log pdf
    # Starting with the first layer which is Gaussian
    eval += log_pdf_multivariate_normal(
        plntree.mu_fun[0].data, plntree.omega_fun[0].data, Z[:, 0, plntree.layer_masks[0]]
    )
    # The propagation is Gaussian Markov
    for layer in range(0, len(plntree.K) - 1):
        Z_prev = Z[:, layer, plntree.layer_masks[layer]]
        Z_cur = Z[:, layer + 1, plntree.layer_masks[layer + 1]]
        mu_cur = plntree.mu_fun[layer + 1](Z_prev)
        omega_cur = plntree.omega_fun[layer + 1](Z_prev)
        eval += log_pdf_multivariate_normal(mu_cur, omega_cur, Z_cur)
    return eval

def clr_transform(X):
    """Center log-ratio transformation."""
    if type(X) == torch.Tensor:
        X_log = torch.log(X + 1e-10)
        X_geometric_mean = X_log.mean(-1, keepdim=True)
        return X_log - X_geometric_mean
    X_log = np.log(X + 1e-10)
    X_geometric_mean = X_log.mean(-1, keepdims=True)
    return X_log - X_geometric_mean

def invert_clr_transform(X_clr):
    """Invert the center log-ratio transformation."""
    X = np.exp(X_clr - X_clr.max(axis=-1, keepdims=True))
    return X / X.sum(axis=-1, keepdims=True)

