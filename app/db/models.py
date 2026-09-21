from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base




class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)

    original_filename: Mapped[str] = mapped_column(String)

    storage_path: Mapped[str] = mapped_column(String)

    row_count: Mapped[int] = mapped_column(Integer)

    column_count: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )



class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
    )

    model_id: Mapped[int | None] = mapped_column(
        ForeignKey("models.id"),
        nullable=True,
    )

    target_column: Mapped[str] = mapped_column(String)

    task_type: Mapped[str] = mapped_column(
        String,
        default="regression",
    )

    status: Mapped[str] = mapped_column(
        String,
        default="queued",
    )

    current_trial: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    total_trials: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    current_fold: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    total_folds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )



class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
    )

    target_column: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    task_type: Mapped[str] = mapped_column(
        String,
        default="regression",
    )

    model_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    preprocessor_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    model_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    training_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    mean_rmse: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    std_rmse: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
