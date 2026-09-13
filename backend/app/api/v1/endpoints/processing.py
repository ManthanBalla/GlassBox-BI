"""Data Processing Agent Endpoints for API v1.

Provides REST interfaces to execute cleaning, imputation, duplicate resolution,
gap analysis, outlier detection, and leakage-safe feature engineering.
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel, Field

from backend.app.schemas.data_contract import ColumnMapping, SYNTHETIC_RETAIL_MAPPING
from backend.app.schemas.processing import DataProcessingConfig, DataProcessingResult
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor

router = APIRouter()
processor = GenericBusinessDataProcessor()


class LocalProcessRequest(BaseModel):
    """Payload for processing a local server-side CSV file."""
    file_path: str = Field(..., description="Path to CSV dataset on the server")
    dataset_name: Optional[str] = Field(default=None, description="Optional custom dataset name")
    dataset_id: Optional[str] = Field(default=None, description="Optional custom identifier")
    mapping: Optional[ColumnMapping] = Field(default=None, description="Optional column mapping")
    config: Optional[DataProcessingConfig] = Field(default=None, description="Processing configuration")


@router.post("/file", response_model=DataProcessingResult, summary="Process Uploaded CSV Dataset")
async def process_dataset_file(
    file: UploadFile = File(..., description="CSV file to process"),
    dataset_name: Optional[str] = Form(default=None),
    dataset_id: Optional[str] = Form(default=None),
) -> DataProcessingResult:
    """Cleans, normalizes, detects outliers, and engineers leakage-safe features for an uploaded CSV."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are currently supported for processing.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(exc)}")

    _, result = processor.process(
        df=df,
        dataset_name=dataset_name or file.filename,
        dataset_id=dataset_id,
    )
    return result


@router.post("/local", response_model=DataProcessingResult, summary="Process Local Server-Side CSV")
def process_local_file(request: LocalProcessRequest) -> DataProcessingResult:
    """Processes a local CSV dataset on the server and returns the machine-readable result payload."""
    path_obj = Path(request.file_path)
    if not path_obj.exists():
        raise HTTPException(status_code=404, detail=f"File not found at path: {request.file_path}")

    try:
        df = pd.read_csv(path_obj)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV at {request.file_path}: {str(exc)}")

    _, result = processor.process(
        df=df,
        mapping=request.mapping,
        config=request.config,
        dataset_id=request.dataset_id,
        dataset_name=request.dataset_name or path_obj.stem,
    )
    return result


@router.get("/config", response_model=DataProcessingConfig, summary="Get Default Processing Configuration")
def get_default_config() -> DataProcessingConfig:
    """Returns the standard default configuration used by the Data Processing Agent."""
    return DataProcessingConfig()


@router.get("/sample-summary", response_model=DataProcessingResult, summary="Process Committed Retail Sample")
def get_sample_processing_summary() -> DataProcessingResult:
    """Executes the processing pipeline on the committed sample dataset ('data/sample/retail_sample.csv')."""
    sample_path = Path("data/sample/retail_sample.csv")
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample dataset not found at 'data/sample/retail_sample.csv'. Please generate it first.",
        )

    df = pd.read_csv(sample_path)
    _, result = processor.process(
        df=df,
        mapping=SYNTHETIC_RETAIL_MAPPING,
        dataset_id="proc_retail_sample_001",
        dataset_name="Committed_Retail_Sample",
    )
    return result
