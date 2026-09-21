import torch

from ml.models import TabularModel


def test_tabular_model_output_shape():
    model = TabularModel(
        num_numeric_features=2,
        categorical_cardinalities=[5],
        embedding_dims=[3],
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
    )

    numeric = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    categorical = torch.tensor([
        [1],
        [2],
    ])

    output = model(
        numeric,
        categorical,
    )

    assert output.shape == (2, 1)