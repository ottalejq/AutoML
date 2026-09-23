from uuid import uuid4

import pytest

from app.db.models import Dataset
from app.services.dataset_service import get_dataset


def test_get_dataset_returns_saved_record(db_session):
    dataset = Dataset(filename="data.csv", path="data.csv", target_column="target")
    db_session.add(dataset)
    db_session.commit()

    result = get_dataset(db=db_session, dataset_id=dataset.id)

    assert result.id == dataset.id
    assert result.filename == "data.csv"
    assert result.target_column == "target"


def test_get_dataset_rejects_unknown_id(db_session):
    with pytest.raises(ValueError, match="Dataset not found"):
        get_dataset(db=db_session, dataset_id=uuid4())
