import torch
from torch.utils.data import TensorDataset, DataLoader
from plntree.utils import seed_all


def numpy_dataset_to_torch_dataloader(X, y, batch_size=32, shuffle=True, onehot=True, seed=None):
    seed_all(seed)
    # Convert NumPy arrays to PyTorch tensors
    data_tensor = torch.tensor(X, dtype=torch.float64)
    if onehot:
        y_ = torch.tensor(y).type(torch.int64)
        n_classes = len(torch.unique(y_))
        labels_tensor = torch.nn.functional.one_hot(y_, n_classes).type(torch.float64)
    else:
        labels_tensor = torch.tensor(y, dtype=torch.int64)

    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
