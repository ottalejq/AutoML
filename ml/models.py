import torch
from torch import nn


class TabularModel(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        embedding_dims: list[int],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim in zip(categorical_cardinalities, embedding_dims)
        ])

        layers = []
        current_dim = num_numeric_features + sum(embedding_dims)
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, numeric, categorical):
        embedded = [
            embedding(categorical[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat([numeric, *embedded], dim=1) if embedded else numeric
        return self.network(x)
