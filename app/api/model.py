from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException


from pydantic import BaseModel
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.model_service import predict_with_model

from app.db.models import Model as DBModel


router = APIRouter(
    prefix="/models",
    tags=["models"],
)




class PredictionRequest(BaseModel):
    rows: list[dict]


@router.post("/{model_id}/predict")
def predict_model(
    model_id: UUID,
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    try:
        predictions = predict_with_model(
            model_id=model_id,
            rows=request.rows,
            db=db,
        )
    except ValueError as exc:
        message = str(exc)

        if message == "Model not found.":
            raise HTTPException(
                status_code=404,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=422,
            detail=message,
        ) from exc
        
    return {
        "model_id": model_id,
        "predictions": predictions,
    }



@router.get("/{model_id}/info")
def model_info(
    model_id: UUID,
    db: Session = Depends(get_db),
):
    model_record = db.get(DBModel, model_id)

    if model_record is None:
        raise HTTPException(status_code=404, detail="Model not found.")

    return {
        column.key: getattr(model_record, column.key)
        for column in inspect(DBModel).column_attrs
    }
