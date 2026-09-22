import numpy as np

from ml.models import NeuralNetworkModel


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

    model = NeuralNetworkModel(
        num_numeric=1,
        cat_cardinalities=[],
        hidden_dim=8,
        n_layers=1,
    )

    before = model.predict((numeric, categorical)).copy()

    result = model.fit(
        X_train=(numeric, categorical),
        y_train=target,
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        device="cpu",
    )

    assert result is model
    predictions = model.predict((numeric, categorical))
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert not np.array_equal(predictions, before)
