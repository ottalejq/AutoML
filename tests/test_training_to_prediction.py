from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.db.models import Dataset
from app.db.models import Model as ModelRecord
from app.services.model_service import predict_with_model
from app.services.training_service import run_training
from ml import pipeline
from ml.config import SEARCH_CONFIG
from ml.models import ElasticNetModel, GradientBoostingModel, NeuralNetworkModel


@pytest.mark.parametrize(
    "model_class,model_params,fit_params",
    [
        (ElasticNetModel, {"alpha": [0.01]}, {}),
        (GradientBoostingModel, {"n_estimators": [5], "min_child_samples": [2]}, {}),
        (
            NeuralNetworkModel,
            {"hidden_dim": [8], "n_layers": [1]},
            {"epochs": [2], "batch_size": [8], "device": ["cpu"]},
        ),
    ],
)
def test_training_to_prediction(
    db_session, tmp_path, monkeypatch, model_class, model_params, fit_params
):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "x1": range(30),
            "city": ["Berlin", "Hamburg"] * 15,
            "target": [x * 3 + 1 for x in range(30)],
        }
    )
    df.to_csv(csv_path, index=False)
    dataset = Dataset(filename="data.csv", path=str(csv_path), target_column="target")
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    monkeypatch.setattr("app.services.model_service.MODEL_DIR", tmp_path)
    monkeypatch.setitem(SEARCH_CONFIG, "n_trials", 1)
    monkeypatch.setitem(SEARCH_CONFIG, "n_folds", 2)
    monkeypatch.setattr(pipeline, "MODEL_SEARCH_SPACES", {model_class: model_params})
    monkeypatch.setattr(pipeline, "FIT_SEARCH_SPACES", {model_class: fit_params})

    result = run_training(dataset_id=dataset.id, db=db_session)
    db_session.expire_all()
    record = db_session.get(ModelRecord, result["model_id"])

    assert record is not None
    assert record.dataset_id == dataset.id
    assert record.model_class == model_class.__name__
    assert record.target_column == "target"
    assert record.metric == SEARCH_CONFIG["metric"]
    assert np.isfinite(record.test_score)
    assert record.test_score == result["test_score"]
    assert Path(record.path) == tmp_path / f"{record.id}.joblib"
    assert Path(record.path).is_file()

    predictions = predict_with_model(
        model_id=record.id,
        rows=[{"x1": 31, "city": "Berlin"}, {"x1": 32, "city": "Hamburg"}],
        db=db_session,
    )

    assert len(predictions) == 2
    assert all(isinstance(value, float) for value in predictions)
    assert np.isfinite(predictions).all()
