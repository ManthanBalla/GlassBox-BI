"""Dataset Ingestion and Data Foundation Endpoints for API v1."""

from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.app.data_processing.ingestion import DatasetIngestionService
from backend.app.schemas.data_contract import (
    ColumnMapping,
    DEFAULT_CANONICAL_MAPPING,
    SYNTHETIC_RETAIL_MAPPING,
    WALMART_MAPPING_TEMPLATE,
    ROSSMANN_MAPPING_TEMPLATE,
)
from backend.app.schemas.ingestion import IngestionResult

router = APIRouter()
ingestion_service = DatasetIngestionService()


class LocalIngestRequest(BaseModel):
    """Payload for ingesting a local server-side CSV file."""
    file_path: str = Field(..., description="Local filepath to the CSV dataset")
    dataset_name: Optional[str] = Field(default=None, description="Optional custom name for the dataset")
    dataset_id: Optional[str] = Field(default=None, description="Optional custom ID")
    mapping: Optional[ColumnMapping] = Field(default=None, description="Optional column mapping specification")


@router.post("/ingest/file", response_model=IngestionResult, summary="Ingest CSV Dataset via Upload")
async def ingest_dataset_file(
    file: UploadFile = File(..., description="CSV file to ingest"),
    dataset_name: Optional[str] = Form(default=None),
    dataset_id: Optional[str] = Form(default=None),
) -> IngestionResult:
    """Ingests an uploaded CSV dataset, executing schema detection, validation, profiling, and quality scoring."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are currently supported for ingestion.")

    contents = await file.read()
    result = ingestion_service.ingest_csv(
        source=contents,
        dataset_name=dataset_name or file.filename,
        dataset_id=dataset_id,
    )
    return result


@router.post("/ingest/local", response_model=IngestionResult, summary="Ingest Local Server-Side CSV")
def ingest_local_file(request: LocalIngestRequest) -> IngestionResult:
    """Ingests a CSV dataset from a local path on the server (e.g. synthetic or benchmark data)."""
    result = ingestion_service.ingest_csv(
        source=request.file_path,
        dataset_name=request.dataset_name,
        dataset_id=request.dataset_id,
        mapping=request.mapping,
    )
    return result


@router.get("/sample-summary", response_model=IngestionResult, summary="Ingest Committed Retail Sample")
def get_sample_summary() -> IngestionResult:
    """Quick verification endpoint that ingests and evaluates the committed retail sample dataset."""
    sample_path = Path("data/sample/retail_sample.csv")
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample dataset not found at 'data/sample/retail_sample.csv'. Please generate it first.",
        )

    result = ingestion_service.ingest_csv(
        source=sample_path,
        dataset_name="Committed_Retail_Sample",
        dataset_id="ds_retail_sample_001",
        mapping=SYNTHETIC_RETAIL_MAPPING,
    )
    return result


@router.get("/mapping-templates", summary="Get Pre-configured Dataset Column Mapping Templates")
def get_mapping_templates() -> Dict[str, Any]:
    """Returns standard pre-configured mapping templates (Synthetic, Walmart, Rossmann, Canonical)."""
    return {
        "templates": {
            "canonical_default": DEFAULT_CANONICAL_MAPPING.model_dump(),
            "synthetic_retail": SYNTHETIC_RETAIL_MAPPING.model_dump(),
            "walmart_benchmark": WALMART_MAPPING_TEMPLATE.model_dump(),
            "rossmann_benchmark": ROSSMANN_MAPPING_TEMPLATE.model_dump(),
        }
    }
