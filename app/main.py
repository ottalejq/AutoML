from contextlib import asynccontextmanager

from fastapi import FastAPI

from redis import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends

from app.core.config import settings
from app.db.session import get_db

from app.api.dataset import router as datasets_router
from app.api.training import router as training_router
from app.api.model import router as prediction_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="AutoML",
    description="API for uploading tabular data, training models, and making predictions.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(prediction_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))

    redis = Redis.from_url(settings.celery_broker_url)
    redis.ping()

    return {"status": "ready"}