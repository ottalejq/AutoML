from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from ml.preprocessing import prepare_data

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.db.session import get_db

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIR = Path("storage/datasets")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    db: Session = Depends(get_db)
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

    dataset = Dataset(
        original_filename=file.filename,
        storage_path=str(file_path),
        row_count=len(df),
        column_count=len(df.columns),
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "rows": dataset.row_count,
        "columns": dataset.column_count,
        "target_column": target_column,
    }