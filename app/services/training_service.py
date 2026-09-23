from uuid import UUID

import pandas as pd

from app.db.models import Model as ModelRecord
from app.services.dataset_service import get_dataset
from app.services.model_service import save_model_artifact
from ml.config import SEARCH_CONFIG
from ml.pipeline import train_pipeline


def run_training(
    dataset_id: UUID,
    db,
):
    dataset = get_dataset(
        db=db,
        dataset_id=dataset_id,
    )

    df = pd.read_csv(dataset.path)

    X = df.drop(columns=[dataset.target_column])
    y = df[dataset.target_column]

    result = train_pipeline(
        X=X,
        y=y,
    )

    model_record = ModelRecord(
        dataset_id=dataset.id,
        target_column=dataset.target_column,
        path="",
        model_class=result["model_class"].__name__,
        model_params=result["model_params"],
        fit_params=result["fit_params"],
        metric=SEARCH_CONFIG["metric"],
        test_score=result["test_score"],
    )

    db.add(model_record)
    db.flush()

    model_record.path = save_model_artifact(
        model=result["model"],
        model_id=model_record.id,
    )

    db.commit()
    db.refresh(model_record)

    return {
        "dataset_id": dataset.id,
        "model_id": model_record.id,
        "target_column": dataset.target_column,
        "test_score": model_record.test_score,
    }