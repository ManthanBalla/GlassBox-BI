"""Pydantic Schemas and Configuration for Data Processing Agent (Phase 3).

Establishes schemas for data processing configuration, audit logging,
outlier profiling, time-series gap analysis, temporal splitting, and
machine-readable processing results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from backend.app.schemas.contracts import DatasetMetadata
from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import QualityScoreBreakdown


class MissingNumericStrategy(str, Enum):
    """Strategy for imputing missing values in numeric business features."""
    MEDIAN = "median"
    FORWARD_FILL = "forward_fill"
    CONSTANT = "constant"
    DROP = "drop"


class MissingCategoricalStrategy(str, Enum):
    """Strategy for handling missing values in categorical business dimensions."""
    MODE = "mode"
    UNKNOWN = "unknown"
    DROP = "drop"


class MissingTargetStrategy(str, Enum):
    """Strategy for handling missing target observations.
    
    Zero Leakage Rule: Target values are NEVER filled from future observations.
    """
    DROP = "drop"
    FLAG = "flag"


class DuplicateStrategy(str, Enum):
    """Policy for resolving duplicate records."""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    FLAG = "flag"


class OutlierMethod(str, Enum):
    """Methodology for identifying demand anomalies and extreme values."""
    IQR = "iqr"
    ZSCORE = "zscore"


class OutlierStrategy(str, Enum):
    """Strategy for handling detected outliers.
    
    In retail business forecasting, spikes caused by promotions and holidays
    represent genuine business events. Default is FLAG rather than DELETE.
    """
    FLAG = "flag"
    WINSORIZE = "winsorize"
    CLIP = "clip"
    REMOVE = "remove"


class DataProcessingConfig(BaseModel):
    """Configuration contract controlling all data processing and feature engineering stages."""
    
    # Missing Value Handling
    missing_numeric_strategy: MissingNumericStrategy = Field(
        default=MissingNumericStrategy.MEDIAN,
        description="Imputation method for numeric attributes (median, forward_fill, constant)",
    )
    numeric_constant_fallback: float = Field(
        default=0.0,
        description="Fallback constant when median/forward fill is inapplicable",
    )
    missing_categorical_strategy: MissingCategoricalStrategy = Field(
        default=MissingCategoricalStrategy.UNKNOWN,
        description="Handling for missing categorical values (mode, unknown)",
    )
    missing_target_strategy: MissingTargetStrategy = Field(
        default=MissingTargetStrategy.DROP,
        description="Policy for missing target records; never imputed from future",
    )

    # Duplicate Handling
    duplicate_strategy: DuplicateStrategy = Field(
        default=DuplicateStrategy.KEEP_FIRST,
        description="Policy for resolving records sharing the same business key",
    )

    # Invalid Business Values
    clip_negative_prices: bool = Field(
        default=True,
        description="Whether to clip or correct non-positive prices",
    )
    clip_negative_inventory: bool = Field(
        default=True,
        description="Whether to clip negative inventory levels to 0.0",
    )

    # Outlier Detection
    outlier_method: OutlierMethod = Field(
        default=OutlierMethod.IQR,
        description="Statistical method for outlier detection (iqr, zscore)",
    )
    outlier_strategy: OutlierStrategy = Field(
        default=OutlierStrategy.FLAG,
        description="Action taken on outliers: flag, winsorize, clip, remove",
    )
    outlier_threshold: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="IQR multiplier (typically 1.5) or Z-score threshold (e.g. 3.0)",
    )
    group_aware_outliers: bool = Field(
        default=True,
        description="Detect outliers within entity/product groups when sufficient data exists",
    )

    # Time-Series Sorting & Gap Integrity
    infer_frequency: bool = Field(
        default=True,
        description="Attempt to infer canonical time-series frequency (e.g. 'D', 'W')",
    )

    # Feature Engineering
    feature_engineering_enabled: bool = Field(
        default=True,
        description="Master switch for automated feature engineering",
    )
    calendar_features_enabled: bool = Field(
        default=True,
        description="Generate calendar attributes: year, month, quarter, day_of_week, etc.",
    )
    lag_features_enabled: bool = Field(
        default=True,
        description="Generate historical lag features for the target variable",
    )
    lags: List[int] = Field(
        default_factory=lambda: [1, 7, 14, 28],
        description="Lag offsets in observation periods (strictly historical)",
    )
    rolling_features_enabled: bool = Field(
        default=True,
        description="Generate rolling statistical windows for target variable",
    )
    rolling_windows: List[int] = Field(
        default_factory=lambda: [7, 14, 28],
        description="Rolling window sizes (applied with shift(1) to avoid lookahead leakage)",
    )
    business_features_enabled: bool = Field(
        default=True,
        description="Generate generic business features: price changes, stockout flags, etc.",
    )

    # Target Preservation
    preserve_original_target: bool = Field(
        default=True,
        description="Preserve raw target as target_original if any target transformation is requested",
    )

    # Chronological Splitting
    train_ratio: float = Field(
        default=0.70,
        ge=0.1,
        le=0.9,
        description="Proportion of chronological timeline allocated to training",
    )
    val_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Proportion allocated to validation",
    )
    test_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Proportion allocated to holdout testing",
    )

    # Reproducibility
    random_seed: int = Field(
        default=42,
        description="Fixed random seed for deterministic sampling if applicable",
    )


class ProcessingAuditEntry(BaseModel):
    """Auditable record documenting an individual data processing transformation."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = Field(..., description="Name of the processing transformation step")
    column: Optional[str] = Field(default=None, description="Target column or dimension affected")
    rows_affected: int = Field(default=0, ge=0, description="Count of observations modified or dropped")
    reason: str = Field(..., description="Justification / trigger for the transformation")
    strategy: str = Field(..., description="Specific algorithmic policy applied")
    before_summary: Optional[Dict[str, Any]] = Field(default=None, description="State prior to transformation")
    after_summary: Optional[Dict[str, Any]] = Field(default=None, description="State following transformation")


class TimeSeriesIntegrityReport(BaseModel):
    """Detailed diagnostic report on time-series temporal continuity and gaps."""
    expected_frequency: Optional[str] = Field(default="D", description="Inferred or expected observation cadence")
    total_series_count: int = Field(default=0, ge=0, description="Number of distinct time-series groups")
    series_with_gaps: int = Field(default=0, ge=0, description="Count of time-series with missing observation periods")
    total_missing_periods: int = Field(default=0, ge=0, description="Sum of all missing time periods across series")
    largest_gap_periods: int = Field(default=0, ge=0, description="Maximum consecutive missing periods in any series")
    continuity_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of expected observations present across the full timeline",
    )
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    min_series_length: int = 0
    max_series_length: int = 0


class OutlierSummary(BaseModel):
    """Statistical summary of detected demand and feature outliers."""
    method: str = Field(..., description="Detection method applied (IQR, Z-Score)")
    strategy: str = Field(..., description="Action taken (flag, clip, winsorize, remove)")
    total_outliers: int = Field(default=0, ge=0, description="Total observations flagged as outliers")
    outlier_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Proportion of dataset flagged")
    affected_entities_count: int = Field(default=0, ge=0, description="Number of entities exhibiting outliers")
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    preservation_rationale: str = Field(
        default="Demand spikes during promotions/holidays are preserved as legitimate business signals.",
        description="Justification for flagging rather than deletion",
    )


class TemporalSplitMetadata(BaseModel):
    """Chronological boundaries and row distributions for walk-forward partitioning."""
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_rows: int
    val_rows: int
    test_rows: int
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    val_start_date: Optional[str] = None
    val_end_date: Optional[str] = None
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    temporal_order_verified: bool = Field(
        default=True,
        description="Verified that train < validation < test without chronological overlap",
    )


class DataProcessingResult(BaseModel):
    """Complete machine-readable result payload of a Data Processing Agent run."""
    dataset_id: str
    dataset_name: str
    status: str = Field(..., description="'SUCCESS', 'WARNING', or 'FAILED'")
    config: DataProcessingConfig
    
    # Row and Column Metrics
    input_row_count: int
    output_row_count: int
    input_column_count: int
    output_column_count: int
    input_columns: List[str]
    output_columns: List[str]
    removed_rows_count: int = 0
    modified_rows_count: int = 0
    
    # Missing & Duplicate Accounting
    missing_values_before: Dict[str, int] = Field(default_factory=dict)
    missing_values_after: Dict[str, int] = Field(default_factory=dict)
    duplicates_before: int = 0
    duplicates_after: int = 0
    conflicting_duplicates_count: int = 0
    
    # Statistical & Temporal Profiling
    outlier_summary: OutlierSummary
    time_series_integrity: TimeSeriesIntegrityReport
    temporal_split: TemporalSplitMetadata
    generated_features: List[str] = Field(default_factory=list)
    
    # Quality Score Tracking
    quality_score_before: QualityScoreBreakdown
    quality_score_after: QualityScoreBreakdown
    
    # Audit Trail & Diagnostics
    audit_trail: List[ProcessingAuditEntry] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_duration_seconds: float = 0.0
    processed_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First few rows of processed dataframe showing generated features",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
