import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.lstm(X)[0][:, -1, :]
        return self.classifier(X)


class DenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_classes, selected_layer=None):
        super(DenseClassifier, self).__init__()
        hidden_sizes = [input_size] + hidden_sizes
        self.classifier = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.ReLU()
            ) for i in range(len(hidden_sizes) - 1)],
            nn.Linear(hidden_sizes[-1], n_classes),
            nn.Softmax(dim=1)
        )
        self.selected_layer = selected_layer

    def forward(self, X):
        if self.selected_layer is not None:
            X = X[:, self.selected_layer]
        return self.classifier(X)
