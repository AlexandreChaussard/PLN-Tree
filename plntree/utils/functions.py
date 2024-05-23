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
    return eval


def log_pdf_pln(X, Z):
    # eval = torch.distributions.Poisson(torch.exp(Z_1)).log_prob(X_1).sum()
    return (Z * X - torch.exp(Z) - torch.lgamma(X + 1)).sum()


def log_pdf_multinomial_plntree(X_parent, X_child, Z_child):
    # for i in range(X.shape[0]):
    #    eval += torch.distributions.Multinomial(
    #        total_count=int(X_parent[i]),
    #        probs=torch.nn.functional.softmax(Z_child[i] - Z_child[i].max()),
    #    ).log_prob(X_child[i].to(torch.int64)).sum()
    eval = (torch.lgamma(X_parent + 1) - torch.lgamma(X_child + 1).sum(-1)).sum()
    eval += (Z_child * X_child).sum()
    eval += (-X_parent * torch.logsumexp(torch.exp(Z_child), dim=-1)).sum()
    return eval
