import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    numeric_data,
    categorical_data,
    target,
):
    model.eval()

    numeric_data = torch.tensor(
        numeric_data, 
        dtype=torch.float32
    )
    categorical_data = torch.tensor(
        categorical_data, 
        dtype=torch.long
    )

    with torch.no_grad():
        predictions = model(
            numeric_data,
            categorical_data,
        ).squeeze(1).numpy()

    mse = mean_squared_error(target, predictions)

    return {
        "mae": mean_absolute_error(target, predictions),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(target, predictions),
    }