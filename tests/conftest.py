import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Allow collection without a local .env or running external services.
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


@pytest.fixture
def db_session():
    from app.db import models  # noqa: F401 -- register tables with Base.metadata
    from app.db.session import Base

    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()
