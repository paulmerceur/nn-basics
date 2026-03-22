import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],   # e.g. [128, 64] for two hidden layers
        activation=nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
        input_size: int = 784,
        output_size: int = 10,
    ):
        super().__init__()

        layers = []
        in_size = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = h

        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))  # flatten (B, 1, 28, 28) → (B, 784)