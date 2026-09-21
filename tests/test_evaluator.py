import numpy as np

from ml.models import TabularModel
from ml.evaluator import evaluate_model


def test_evaluator_returns_metrics():
    numeric = np.array([
        [1.0],
        [2.0],
        [3.0],
    ], dtype=np.float32)

    categorical = np.empty(
        (3, 0),
        dtype=np.int64,
    )

    target = np.array([
        2.0,
        4.0,
        6.0,
    ])

    model = TabularModel(
        num_numeric_features=1,
        categorical_cardinalities=[],
        embedding_dims=[],
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    metrics = evaluate_model(
        model=model,
        numeric_data=numeric,
        categorical_data=categorical,
        target=target,
    )

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics