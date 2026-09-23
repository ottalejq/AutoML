import torch

from ml.models import TabularNetwork


def test_tabular_model_output_shape():
    model = TabularNetwork(
        num_numeric=2,
        cat_cardinalities=[5],
        embedding_dim=3,
        hidden_dim=16,
        n_layers=2,
    )

    numeric = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    categorical = torch.tensor(
        [
            [1],
            [2],
        ]
    )

    output = model(
        numeric,
        categorical,
    )

    assert output.shape == (2, 1)
