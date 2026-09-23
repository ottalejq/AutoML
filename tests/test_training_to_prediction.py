import pandas as pd

from app.db.models import Dataset
from app.services.model_service import predict_with_model
from app.services.training_service import run_training
from ml.config import SEARCH_CONFIG


def test_training_to_prediction(db_session, tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"

    df = pd.DataFrame(
        {
            "x1": range(30),
            "x2": [x * 2 for x in range(30)],
            "target": [x * 3 + 1 for x in range(30)],
        }
    )
    df.to_csv(csv_path, index=False)

    dataset = Dataset(
        filename="data.csv",
        path=str(csv_path),
        target_column="target",
    )

    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    monkeypatch.setitem(SEARCH_CONFIG, "n_trials", 1)
    monkeypatch.setitem(SEARCH_CONFIG, "n_folds", 2)

    result = run_training(
        dataset_id=dataset.id,
        db=db_session,
    )

    predictions = predict_with_model(
        model_id=result["model_id"],
        rows=[
            {
                "x1": 31,
                "x2": 62,
            }
        ],
        db=db_session,
    )

    assert len(predictions) == 1
    assert isinstance(predictions[0], float)