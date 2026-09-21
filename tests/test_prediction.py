import pandas as pd

from ml.models import TabularModel
from ml.preprocessing import TabularPreprocessor
from ml.prediction import predict


def test_prediction_returns_one_value_per_row():
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0],
    })

    preprocessor = TabularPreprocessor()
    preprocessor.fit(df)

    model = TabularModel(
        num_numeric_features=1,
        categorical_cardinalities=[],
        embedding_dims=[],
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    rows = [
        {"age": 25.0},
        {"age": 35.0},
    ]

    predictions = predict(
        model=model,
        preprocessor=preprocessor,
        rows=rows,
    )

    assert len(predictions) == 2