import pandas as pd

from app.db.models import Dataset
from ml.preprocessing import prepare_data
from ml.models import build_model
from ml.trainer import train_model
from ml.evaluator import evaluate_model


def run_training(
    dataset: Dataset,
    target_column: str,
):
    df = pd.read_csv(dataset.storage_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist"
        )

    prepared_data = prepare_data(
        df=df,
        target_column=target_column,
    )

    model = build_model(prepared_data)

    trained_model = train_model(
        model=model,
        prepared_data=prepared_data,
    )

    metrics = evaluate_model(
        model=trained_model,
        prepared_data=prepared_data,
    )

    return {
        "dataset_id": dataset.id,
        "target_column": target_column,
        "metrics": metrics,
    }