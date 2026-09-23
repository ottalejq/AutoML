from pathlib import Path
from uuid import UUID

import pandas as pd

from app.db.models import Model as DBModel
from ml.models import Model

MODEL_DIR = Path("storage/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model_artifact(
    model: Model,
    model_id: UUID,
) -> str:
    path = MODEL_DIR / f"{model_id}.joblib"

    model.save(path)

    return str(path)


def predict_with_model(
    model_id: UUID,
    rows: list[dict],
    db,
):
    model_record = db.get(
        DBModel,
        model_id,
    )

    if model_record is None:
        raise ValueError("Model not found.")

    model = Model.load(Path(model_record.path))

    X = pd.DataFrame(rows)

    expected_columns = model.preprocessor.num_cols + model.preprocessor.cat_cols

    missing = set(expected_columns) - set(X.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return model.predict(X).tolist()
