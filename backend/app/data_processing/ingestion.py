"""Source-Agnostic Dataset Ingestion Service for GlassBox-BI.

Orchestrates CSV loading, column mapping, canonical contract transformation,
validation, temporal leakage detection, profiling, and quality scoring.
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from backend.app.schemas.contracts import DatasetMetadata
from backend.app.schemas.data_contract import (
    BusinessTimeSeriesRecord,
    ColumnMapping,
    auto_detect_column_mapping,
)
from backend.app.schemas.ingestion import (
    DatasetProfile,
    IngestionResult,
    QualityScoreBreakdown,
    TemporalLeakageReport,
    ValidationIssue,
    ValidationSeverity,
)
from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector
from backend.app.data_processing.quality import DataQualityScorer
from backend.app.data_processing.profiling import DataProfiler


class DatasetIngestionService:
    """Enterprise-grade, source-agnostic dataset ingestion service."""

    def __init__(
        self,
        validator: Optional[DataValidator] = None,
        leakage_detector: Optional[TemporalLeakageDetector] = None,
        quality_scorer: Optional[DataQualityScorer] = None,
        profiler: Optional[DataProfiler] = None,
    ):
        self.validator = validator or DataValidator()
        self.leakage_detector = leakage_detector or TemporalLeakageDetector()
        self.quality_scorer = quality_scorer or DataQualityScorer()
        self.profiler = profiler or DataProfiler()

    def ingest_csv(
        self,
        source: Union[str, Path, io.BytesIO, io.StringIO, bytes],
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        mapping: Optional[ColumnMapping] = None,
    ) -> IngestionResult:
        """Ingests a CSV dataset from a filepath, raw bytes, or IO stream.

        Args:
            source: Path to CSV or buffer/bytes containing CSV content.
            dataset_name: Optional human-readable name for the dataset.
            dataset_id: Optional unique identifier (generated if omitted).
            mapping: Optional dataset-specific column mapping (auto-detected if omitted).

        Returns:
            IngestionResult containing complete metadata, validation, quality, and profile.
        """
        now = datetime.now(timezone.utc)
        resolved_name = dataset_name or "Unnamed_Dataset"
        resolved_id = dataset_id or f"ds_{now.strftime('%Y%m%d_%H%M%S')}"

        # 1. Load CSV with graceful error handling
        df: Optional[pd.DataFrame] = None
        load_error: Optional[str] = None

        try:
            if isinstance(source, (str, Path)):
                path_obj = Path(source)
                if not path_obj.exists():
                    load_error = f"Source file does not exist: {path_obj}"
                else:
                    resolved_name = dataset_name or path_obj.stem
                    df = pd.read_csv(path_obj)
            elif isinstance(source, bytes):
                df = pd.read_csv(io.BytesIO(source))
            elif isinstance(source, (io.BytesIO, io.StringIO)):
                df = pd.read_csv(source)
            else:
                load_error = f"Unsupported source type: {type(source)}"
        except pd.errors.EmptyDataError:
            load_error = "The provided CSV file is completely empty."
        except pd.errors.ParserError as pe:
            load_error = f"CSV parsing error: {str(pe)}"
        except Exception as exc:
            load_error = f"Unexpected error reading dataset: {str(exc)}"

        if load_error or df is None:
            return self._build_failed_result(
                dataset_id=resolved_id,
                dataset_name=resolved_name,
                error_message=load_error or "Unknown failure loading dataset.",
                mapping=mapping or ColumnMapping(),
            )

        # 2. Resolve Column Mapping
        effective_mapping = mapping if mapping is not None else auto_detect_column_mapping(list(df.columns))

        # 3. Execute Multi-Dimensional Validation
        errors, warnings = self.validator.validate(df, effective_mapping)

        # 4. Detect Temporal Lookahead Leakage
        leakage_report = self.leakage_detector.detect(df, effective_mapping)

        # 5. Profile Statistical Distributions
        profile = self.profiler.profile(df, effective_mapping)

        # 6. Compute Explainable Quality Score
        quality_score = self.quality_scorer.score(
            df=df,
            mapping=effective_mapping,
            validation_issues=errors + warnings,
            leakage_report=leakage_report,
        )

        # 7. Convert Head Rows to Canonical Contract Sample
        canonical_sample = self._extract_canonical_sample(df, effective_mapping, max_rows=5)

        # 8. Determine Overall Status
        if len(errors) > 0:
            status = "FAILED"
        elif len(warnings) > 0 or leakage_report.has_leakage or quality_score.overall_score < 80.0:
            status = "WARNING"
        else:
            status = "SUCCESS"

        # 9. Assemble DatasetMetadata
        metadata = DatasetMetadata(
            dataset_id=resolved_id,
            name=resolved_name,
            filename=str(source) if isinstance(source, (str, Path)) else f"{resolved_name}.csv",
            row_count=profile.row_count,
            column_count=profile.column_count,
            timestamp_column=effective_mapping.date if effective_mapping.date in df.columns else None,
            target_column=effective_mapping.target if effective_mapping.target in df.columns else None,
            frequency=profile.date_frequency,
            created_at=now,
        )

        return IngestionResult(
            dataset_id=resolved_id,
            dataset_name=resolved_name,
            status=status,
            source_type="CSV",
            metadata=metadata,
            column_mapping=effective_mapping,
            detected_schema=profile.data_types,
            validation_errors=errors,
            validation_warnings=warnings,
            quality_score=quality_score,
            profiling_summary=profile,
            leakage_report=leakage_report,
            canonical_sample=canonical_sample,
            created_at=now,
        )

    def _extract_canonical_sample(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        max_rows: int = 5,
    ) -> List[Dict[str, Any]]:
        """Transforms the top N rows into canonical dictionary representations."""
        sample_rows: List[Dict[str, Any]] = []
        head_df = df.head(max_rows)

        # Check required columns
        has_reqs = (
            mapping.date in head_df.columns
            and mapping.entity_id in head_df.columns
            and mapping.target in head_df.columns
        )
        if not has_reqs:
            return sample_rows

        for _, row in head_df.iterrows():
            try:
                record = {
                    "date": str(row[mapping.date]),
                    "entity_id": str(row[mapping.entity_id]),
                    "target": float(row[mapping.target]),
                    "product_id": str(row[mapping.product_id]) if mapping.product_id and mapping.product_id in head_df.columns and pd.notna(row[mapping.product_id]) else None,
                    "category": str(row[mapping.category]) if mapping.category and mapping.category in head_df.columns and pd.notna(row[mapping.category]) else None,
                    "region": str(row[mapping.region]) if mapping.region and mapping.region in head_df.columns and pd.notna(row[mapping.region]) else None,
                    "store_type": str(row[mapping.store_type]) if mapping.store_type and mapping.store_type in head_df.columns and pd.notna(row[mapping.store_type]) else None,
                    "price": float(row[mapping.price]) if mapping.price and mapping.price in head_df.columns and pd.notna(row[mapping.price]) else None,
                    "promotion": int(row[mapping.promotion]) if mapping.promotion and mapping.promotion in head_df.columns and pd.notna(row[mapping.promotion]) else None,
                    "holiday": int(row[mapping.holiday]) if mapping.holiday and mapping.holiday in head_df.columns and pd.notna(row[mapping.holiday]) else None,
                    "inventory": float(row[mapping.inventory]) if mapping.inventory and mapping.inventory in head_df.columns and pd.notna(row[mapping.inventory]) else None,
                }
                sample_rows.append(record)
            except Exception:
                continue

        return sample_rows

    def _build_failed_result(
        self,
        dataset_id: str,
        dataset_name: str,
        error_message: str,
        mapping: ColumnMapping,
    ) -> IngestionResult:
        """Constructs a graceful failed IngestionResult for unparseable or missing inputs."""
        now = datetime.now(timezone.utc)
        error_issue = ValidationIssue(
            issue_type="INGESTION_LOAD_FAILED",
            message=error_message,
            severity=ValidationSeverity.ERROR,
        )
        empty_profile = DatasetProfile(
            row_count=0,
            column_count=0,
            columns=[],
            data_types={},
        )
        empty_quality = QualityScoreBreakdown(
            schema_score=0.0,
            missing_data_score=0.0,
            duplicate_score=0.0,
            temporal_integrity_score=0.0,
            numeric_validity_score=0.0,
            consistency_score=0.0,
            overall_score=0.0,
            grade="F",
            deductions=[{"dimension": "General", "penalty": 100.0, "reason": error_message}],
        )
        empty_leakage = TemporalLeakageReport(
            has_leakage=False,
            warnings=[],
            splitting_recommendation="Ensure dataset is readable before splitting.",
        )
        empty_meta = DatasetMetadata(
            dataset_id=dataset_id,
            name=dataset_name,
            row_count=0,
            column_count=0,
            created_at=now,
        )

        return IngestionResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            status="FAILED",
            source_type="CSV",
            metadata=empty_meta,
            column_mapping=mapping,
            detected_schema={},
            validation_errors=[error_issue],
            validation_warnings=[],
            quality_score=empty_quality,
            profiling_summary=empty_profile,
            leakage_report=empty_leakage,
            canonical_sample=[],
            created_at=now,
        )
