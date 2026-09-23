from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Dataset


def get_dataset(
    db: Session,
    dataset_id: UUID,
) -> Dataset:
    dataset = db.get(
        Dataset,
        dataset_id,
    )

    if dataset is None:
        raise ValueError("Dataset not found.")

    return dataset
