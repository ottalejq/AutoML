from io import BytesIO
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.db.session import get_db

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)

UPLOAD_DIR = Path("storage/datasets")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    target_column: Annotated[str, Form()],
    db: Annotated[Session, Depends(get_db)],
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed.",
        )

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file.",
        ) from exc

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist.",
        )

    dataset = Dataset(
        filename=file.filename,
        path="",
        target_column=target_column,
    )

    db.add(dataset)
    db.flush()

    file_path = UPLOAD_DIR / f"{dataset.id}.csv"

    try:
        file_path.write_bytes(content)

        dataset.path = str(file_path)

        db.commit()
        db.refresh(dataset)

    except Exception:
        db.rollback()

        if file_path.exists():
            file_path.unlink()

        raise

    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "target_column": dataset.target_column,
    }
