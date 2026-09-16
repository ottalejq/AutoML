from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.db.session import get_db
from app.services.training_service import run_training


router = APIRouter(
    prefix="/training",
    tags=["training"],
)


class TrainingRequest(BaseModel):
    dataset_id: int
    target_column: str


@router.post("/")
def train_model(
    request: TrainingRequest,
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, request.dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found",
        )

    result = run_training(
        dataset=dataset,
        target_column=request.target_column,
    )

    return result