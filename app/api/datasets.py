from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from ml.preprocessing import prepare_data


router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIR = Path("storage/datasets")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file"
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist"
        )

    dataset_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{dataset_id}.csv"

    file_path.write_bytes(content)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "target_column": target_column,
        "rows": len(df),
        "columns": len(df.columns),
    }