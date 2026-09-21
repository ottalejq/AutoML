from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.model_service import predict_with_model


router = APIRouter(
    prefix="/models",
    tags=["models"],
)


class PredictionRequest(BaseModel):
    rows: list[dict]


@router.post("/{model_id}/predict")
def predict_model(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    predictions = predict_with_model(
        model_id=model_id,
        rows=request.rows,
        db=db,
    )

    return {
        "model_id": model_id,
        "predictions": predictions,
    }