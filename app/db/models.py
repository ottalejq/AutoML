from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    filename: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    target_column: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
    )

    model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("models.id"),
    )

    target_column: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String,
        default="queued",
    )

    error_message: Mapped[str | None] = mapped_column(
        String,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
    )

    target_column: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    model_class: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    model_params: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    fit_params: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    metric: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    test_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )