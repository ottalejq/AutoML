from datetime import datetime, timezone
from uuid import UUID

from app.db.models import TrainingJob
from app.db.session import SessionLocal
from app.services.training_service import run_training
from worker.celery_app import celery_app


@celery_app.task
def run_training_task(job_id: str):
    db = SessionLocal()

    try:
        job = db.get(
            TrainingJob,
            UUID(job_id),
        )

        if job is None:
            raise ValueError("Training job not found.")

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)

        db.commit()

        result = run_training(
            dataset_id=job.dataset_id,
            db=db,
        )

        job.status = "completed"
        job.model_id = result["model_id"]
        job.completed_at = datetime.now(timezone.utc)

        db.commit()

        return {
            "job_id": str(job.id),
            "model_id": str(job.model_id),
            "status": job.status,
        }

    except Exception as exc:
        db.rollback()

        job = db.get(
            TrainingJob,
            UUID(job_id),
        )

        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.now(timezone.utc)

            db.commit()

        raise

    finally:
        db.close()