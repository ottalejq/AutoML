import json
from pathlib import Path

import joblib
import torch

from app.db.models import Model
from ml.prediction import load_model_and_preprocessor, predict



def save_model_artifacts(
    model,
    preprocessor,
    dataset_id,
    model_config,
    training_config,
    target_column,
):
    model_dir = Path("storage/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_{dataset_id}.pt"
    preprocessor_path = (
        model_dir / f"preprocessor_{dataset_id}.joblib"
    )
    metadata_path = (
        model_dir / f"metadata_{dataset_id}.json"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    metadata = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "model_config": model_config,
        "training_config": training_config,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "metadata_path": str(metadata_path),
    }



def predict_with_model(
    model_id: int,
    rows: list[dict],
    db,
):
    model_record = db.get(Model, model_id)

    if model_record is None:
        raise ValueError("Model not found")

    model, preprocessor = load_model_and_preprocessor(
        model_path=model_record.model_path,
        preprocessor_path=model_record.preprocessor_path,
        model_config=model_record.model_config,
    )

    return predict(
        model=model,
        preprocessor=preprocessor,
        rows=rows,
    )