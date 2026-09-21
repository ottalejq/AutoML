import numpy as np

from ml.models import TabularModel
from ml.trainer import train_model


def test_train_model_runs():
    numeric = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ], dtype=np.float32)

    categorical = np.empty(
        (4, 0),
        dtype=np.int64,
    )

    target = np.array(
        [2.0, 4.0, 6.0, 8.0],
        dtype=np.float32,
    )

    model = TabularModel(
        num_numeric_features=1,
        categorical_cardinalities=[],
        embedding_dims=[],
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    result = train_model(
        model=model,
        numeric_data=numeric,
        categorical_data=categorical,
        target=target,
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
    )

    assert result is model