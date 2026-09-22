from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Dataset, TrainingJob
from app.db.session import get_db
from worker.tasks import run_training_task


router = APIRouter(
    prefix="/training",
    tags=["training"],
)


class TrainingRequest(BaseModel):
    dataset_id: UUID


@router.post("/")
def train_model(
    request: TrainingRequest,
    db: Session = Depends(get_db),
):
    dataset = db.get(
        Dataset,
        request.dataset_id,
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    job = TrainingJob(
        dataset_id=dataset.id,
        target_column=dataset.target_column,
        status="queued",
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    run_training_task.delay(str(job.id))
    
    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "target_column": job.target_column,
        "status": job.status,
    }


@router.get("/{job_id}")
def get_training_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    job = db.get(TrainingJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Training job not found.",
        )

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "model_id": job.model_id,
        "target_column": job.target_column,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
    }