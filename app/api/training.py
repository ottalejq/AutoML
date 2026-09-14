from fastapi import APIRouter, Form

from ml.trainer import run_training

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/")
async def train_model(
    dataset_id: str = Form(...),
    target_column: str = Form(...),
):
    return run_training(dataset_id=dataset_id, target_column=target_column)