from fastapi import FastAPI
from app.api.datasets import router as datasets_router
from app.api.training import router as training_router
from app.api.prediction import router as prediction_router


app = FastAPI(
    title="AutoML",
    description="API for uploading tabular data, training PyTorch models, and making predictions.",
    version="0.1.0",
)

app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(prediction_router)

@app.get("/health")
def health():
    return {"status": "ok"}