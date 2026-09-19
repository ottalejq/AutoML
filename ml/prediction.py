import joblib
import pandas as pd
import torch

from ml.models import TabularModel


def load_model_and_preprocessor(
    model_path,
    preprocessor_path,
    model_config,
):
    preprocessor = joblib.load(preprocessor_path)

    cardinalities = [
        len(preprocessor.category_maps[column]) + 1
        for column in preprocessor.categorical_columns
    ]

    embedding_dims = [
        min(16, max(2, cardinality // 2))
        for cardinality in cardinalities
    ]

    model = TabularModel(
        num_numeric_features=len(
            preprocessor.numeric_columns
        ),
        categorical_cardinalities=cardinalities,
        embedding_dims=embedding_dims,
        **model_config,
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        )
    )

    model.eval()

    return model, preprocessor


def predict(
    model,
    preprocessor,
    rows: list[dict],
):
    df = pd.DataFrame(rows)

    transformed = preprocessor.transform(df)

    numeric = torch.tensor(
        transformed["numeric"],
        dtype=torch.float32,
    )

    categorical = torch.tensor(
        transformed["categorical"],
        dtype=torch.long,
    )

    with torch.no_grad():
        predictions = model(
            numeric,
            categorical,
        ).squeeze(1)

    return predictions.tolist()