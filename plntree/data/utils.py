import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from plntree.utils import seed_all
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def numpy_dataset_to_torch_dataloader(X, y, batch_size=32, shuffle=True, onehot=True, stratified=False, seed=None):
    seed_all(seed)
    # Convert NumPy arrays to PyTorch tensors
    data_tensor = torch.tensor(X, dtype=torch.float64)
    if np.unique(y).shape[0] == 1:
        labels_tensor = torch.zeros((len(y), 1))
    elif onehot and len(y.shape) == 1:
        y_ = torch.tensor(y).type(torch.int64)
        n_classes = len(torch.unique(y_))
        labels_tensor = torch.nn.functional.one_hot(y_, n_classes).type(torch.float64)
    else:
        labels_tensor = torch.tensor(y, dtype=torch.int64)

    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)
    if stratified:
        weights = labels_tensor.sum(dim=0) / len(labels_tensor)
        sampler = WeightedRandomSampler(weights, batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_test(X, y, train_indexes, test_indexes):
    if type(y) == pd.DataFrame or type(y) == pd.Series:
        X_train, y_train = X[train_indexes], y.iloc[train_indexes]
        X_test, y_test = X[test_indexes], y.iloc[test_indexes]
    else:
        X_train, y_train = X[train_indexes], y[train_indexes]
        X_test, y_test = X[test_indexes], y[test_indexes]
    return X_train, y_train, X_test, y_test


def torch_kfold(X, y, batch_size=512, n_repeats=10, train_size=0.8, seed=None):
    seed_all(seed)
    dataloaders = []
    for i, (train_indexes, test_indexes) in enumerate(
            StratifiedShuffleSplit(n_splits=n_repeats, train_size=train_size).split(X, y)):
        X_train, y_train, X_test, y_test = _train_test(X, y, train_indexes, test_indexes)
        assert ((np.unique(y_train) == np.unique(y_test)).all())
        dataloader = numpy_dataset_to_torch_dataloader(X_train, y_train, batch_size=batch_size, stratified=False)
        dataloaders.append((dataloader, X_test, y_test))
    return dataloaders
