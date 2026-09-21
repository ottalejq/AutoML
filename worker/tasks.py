from datetime import datetime

from worker.celery_app import celery_app
from app.db.models import Dataset, TrainingJob
from app.db.session import SessionLocal
from app.services.training_service import run_training


@celery_app.task
def run_training_task(job_id: int):
    db = SessionLocal()

    try:
        job = db.get(TrainingJob, job_id)

        if job is None:
            raise ValueError("Training job not found")

        dataset = db.get(Dataset, job.dataset_id)

        if dataset is None:
            raise ValueError("Dataset not found")

        job.status = "running"
        job.started_at = datetime.utcnow()

        db.commit()

        def update_progress(
            current_trial,
            total_trials,
            current_fold,
            total_folds,
        ):
            job.current_trial = current_trial
            job.total_trials = total_trials
            job.current_fold = current_fold
            job.total_folds = total_folds

            db.commit()

        result = run_training(
            dataset=dataset,
            target_column=job.target_column,
            db=db,
            progress_callback=update_progress,
        )

        job.status = "completed"
        job.model_id = result["model_id"]
        job.completed_at = datetime.utcnow()

        db.commit()

        return {
            "job_id": job.id,
            "model_id": job.model_id,
            "status": job.status,
        }

    except Exception as exc:
        db.rollback()

        job = db.get(TrainingJob, job_id)

        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()

            db.commit()

        raise

    finally:
        db.close()