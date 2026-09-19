from fastapi import APIRouter
from pydantic import BaseModel

from app.services.model_service import predict_with_model


router = APIRouter(
    prefix="/models",
    tags=["models"],
)


class PredictionRequest(BaseModel):
    rows: list[dict]


@router.post("/{dataset_id}/predict")
def predict_model(
    dataset_id: int,
    request: PredictionRequest,
):
    predictions = predict_with_model(
        dataset_id=dataset_id,
        rows=request.rows,
    )

    return {
        "predictions": predictions,
    }