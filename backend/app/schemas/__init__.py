"""GlassBox-BI Pydantic Schemas and API Contracts."""

from backend.app.schemas.health import HealthResponse
from backend.app.schemas.contracts import (
    DatasetMetadata,
    ForecastRequest,
    ForecastResult,
    ExplanationResult,
    RecommendationResult,
)
from backend.app.schemas.data_contract import (
    BusinessTimeSeriesRecord,
    ColumnMapping,
    DEFAULT_CANONICAL_MAPPING,
    SYNTHETIC_RETAIL_MAPPING,
    WALMART_MAPPING_TEMPLATE,
    ROSSMANN_MAPPING_TEMPLATE,
    auto_detect_column_mapping,
)
from backend.app.schemas.ingestion import (
    ValidationSeverity,
    ValidationIssue,
    QualityScoreBreakdown,
    TemporalLeakageReport,
    NumericSummary,
    DatasetProfile,
    IngestionResult,
)

__all__ = [
    "HealthResponse",
    "DatasetMetadata",
    "ForecastRequest",
    "ForecastResult",
    "ExplanationResult",
    "RecommendationResult",
    "BusinessTimeSeriesRecord",
    "ColumnMapping",
    "DEFAULT_CANONICAL_MAPPING",
    "SYNTHETIC_RETAIL_MAPPING",
    "WALMART_MAPPING_TEMPLATE",
    "ROSSMANN_MAPPING_TEMPLATE",
    "auto_detect_column_mapping",
    "ValidationSeverity",
    "ValidationIssue",
    "QualityScoreBreakdown",
    "TemporalLeakageReport",
    "NumericSummary",
    "DatasetProfile",
    "IngestionResult",
]
