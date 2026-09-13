"""Schemas for Dataset Ingestion, Validation, Profiling, and Quality Assessment."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.contracts import DatasetMetadata
from backend.app.schemas.data_contract import ColumnMapping


class ValidationSeverity(str, Enum):
    """Severity classification for validation findings."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationIssue(BaseModel):
    """Standardized representation of a data validation issue."""
    field: Optional[str] = Field(default=None, description="Affected column or entity")
    issue_type: str = Field(..., description="Category code of the issue")
    message: str = Field(..., description="Human-readable explanation of the issue")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING, description="Issue severity level")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic payload / offending values")


class QualityScoreBreakdown(BaseModel):
    """Transparent, deterministic data quality evaluation breakdown (0-100)."""
    schema_score: float = Field(..., ge=0.0, le=100.0, description="Schema completeness and naming quality")
    missing_data_score: float = Field(..., ge=0.0, le=100.0, description="Completeness of records (lack of nulls)")
    duplicate_score: float = Field(..., ge=0.0, le=100.0, description="Freedom from exact and entity-date duplicates")
    temporal_integrity_score: float = Field(..., ge=0.0, le=100.0, description="Continuity, sorting, and frequency regularity")
    numeric_validity_score: float = Field(..., ge=0.0, le=100.0, description="Adherence to non-negative bounds and ranges")
    consistency_score: float = Field(..., ge=0.0, le=100.0, description="Category cardinality and dimension stability")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted composite data quality score")
    grade: str = Field(..., description="Letter grade (A+, A, B, C, D, F)")
    deductions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Explainable audit log explaining every deducted point",
    )


class TemporalLeakageReport(BaseModel):
    """Audit report detecting potential temporal lookahead data leakage."""
    has_leakage: bool = Field(default=False, description="Whether potential temporal leakage was detected")
    flagged_columns: List[str] = Field(default_factory=list, description="Columns flagged with future-derived information")
    warnings: List[str] = Field(default_factory=list, description="Descriptive leakage warnings")
    splitting_recommendation: str = Field(
        default="Use chronological walk-forward train/validation/test splitting for forecasting.",
        description="Prescribed splitting strategy to avoid temporal leakage",
    )


class NumericSummary(BaseModel):
    """Statistical summary for numeric variables."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    outliers_iqr: int = Field(default=0, description="Count of observations exceeding 1.5 * IQR bounds")


class DatasetProfile(BaseModel):
    """Structural and statistical profile of an ingested dataset."""
    row_count: int
    column_count: int
    columns: List[str]
    data_types: Dict[str, str]
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_frequency: Optional[str] = None
    unique_entities: int = 0
    unique_products: int = 0
    unique_categories: int = 0
    unique_regions: int = 0
    missing_percentages: Dict[str, float] = Field(default_factory=dict)
    numeric_summary: Dict[str, NumericSummary] = Field(default_factory=dict)
    target_summary: Optional[NumericSummary] = None


class IngestionResult(BaseModel):
    """Complete machine-readable result payload of a dataset ingestion run."""
    dataset_id: str
    dataset_name: str
    status: str = Field(..., description="'SUCCESS', 'WARNING', or 'FAILED'")
    source_type: str = Field(default="CSV")
    metadata: DatasetMetadata
    column_mapping: ColumnMapping
    detected_schema: Dict[str, str]
    validation_errors: List[ValidationIssue] = Field(default_factory=list)
    validation_warnings: List[ValidationIssue] = Field(default_factory=list)
    quality_score: QualityScoreBreakdown
    profiling_summary: DatasetProfile
    leakage_report: TemporalLeakageReport
    canonical_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First few rows converted to canonical contract format",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
