import torch
from torch import nn


class TabularModel(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        embedding_dims: list[int],
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim
            in zip(categorical_cardinalities, embedding_dims)
        ])

        total_embedding_dim = sum(embedding_dims)

        input_dim = num_numeric_features + total_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, numeric, categorical):
        embedded = [
            embedding(categorical[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        if embedded:
            x = torch.cat([numeric, *embedded], dim=1)
        else:
            x = numeric

        return self.network(x)