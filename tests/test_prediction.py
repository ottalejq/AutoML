import numpy as np
import pandas as pd

from ml.models import Model, NeuralNetworkModel


def test_prediction_returns_one_value_per_row():
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    target = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])

    model = Model(NeuralNetworkModel, hidden_dim=8, n_layers=1)
    model.fit(df, target, epochs=2, batch_size=2, device="cpu")

    rows = pd.DataFrame(
        [
            {"age": 25.0},
            {"age": 35.0},
        ]
    )

    predictions = model.predict(rows)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
