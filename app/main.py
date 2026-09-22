from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.datasets import router as datasets_router
from app.api.training import router as training_router
from app.api.prediction import router as prediction_router
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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