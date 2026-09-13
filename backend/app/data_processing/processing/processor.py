"""Generic Business Data Processor for GlassBox-BI (Phase 3).

Orchestrates the complete data processing pipeline:
Cleaning -> Duplicate Resolution -> Invalid Value Rectification ->
Missing Value Imputation -> Chronological Sorting -> Gap Integrity ->
Outlier Profiling -> Leakage-Safe Feature Engineering ->
Quality Score Evaluation -> Temporal Splitting -> Audit Log Generation.

Strict Rule: Organization-agnostic; operates on canonical contract.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.app.schemas.contracts import DatasetMetadata
from backend.app.schemas.data_contract import (
    ColumnMapping,
    auto_detect_column_mapping,
)
from backend.app.schemas.ingestion import (
    QualityScoreBreakdown,
    TemporalLeakageReport,
    ValidationIssue,
)
from backend.app.schemas.processing import (
    DataProcessingConfig,
    DataProcessingResult,
    OutlierSummary,
    ProcessingAuditEntry,
    TemporalSplitMetadata,
    TimeSeriesIntegrityReport,
)
from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector
from backend.app.data_processing.quality import DataQualityScorer
from backend.app.data_processing.processing.audit import AuditTrailTracker
from backend.app.data_processing.processing.cleaning import DataCleaner
from backend.app.data_processing.processing.duplicates import DuplicateHandler
from backend.app.data_processing.processing.invalid_values import InvalidValueHandler
from backend.app.data_processing.processing.missing_values import MissingValueHandler
from backend.app.data_processing.processing.outliers import OutlierDetector
from backend.app.data_processing.processing.time_series import TimeSeriesAnalyzer
from backend.app.data_processing.processing.feature_engineering import FeatureEngineer
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter


class GenericBusinessDataProcessor:
    """Universal, organization-agnostic data processor for business time-series."""

    def __init__(
        self,
        validator: Optional[DataValidator] = None,
        leakage_detector: Optional[TemporalLeakageDetector] = None,
        quality_scorer: Optional[DataQualityScorer] = None,
        cleaner: Optional[DataCleaner] = None,
        duplicate_handler: Optional[DuplicateHandler] = None,
        invalid_handler: Optional[InvalidValueHandler] = None,
        missing_handler: Optional[MissingValueHandler] = None,
        outlier_detector: Optional[OutlierDetector] = None,
        ts_analyzer: Optional[TimeSeriesAnalyzer] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        splitter: Optional[TimeSeriesSplitter] = None,
    ) -> None:
        self.validator = validator or DataValidator()
        self.leakage_detector = leakage_detector or TemporalLeakageDetector()
        self.quality_scorer = quality_scorer or DataQualityScorer()
        self.cleaner = cleaner or DataCleaner()
        self.duplicate_handler = duplicate_handler or DuplicateHandler()
        self.invalid_handler = invalid_handler or InvalidValueHandler()
        self.missing_handler = missing_handler or MissingValueHandler()
        self.outlier_detector = outlier_detector or OutlierDetector()
        self.ts_analyzer = ts_analyzer or TimeSeriesAnalyzer()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.splitter = splitter or TimeSeriesSplitter()

    def process(
        self,
        df: pd.DataFrame,
        mapping: Optional[ColumnMapping] = None,
        config: Optional[DataProcessingConfig] = None,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, DataProcessingResult]:
        """Executes the complete end-to-end data processing pipeline.

        Args:
            df: Input canonical or raw tabular dataframe.
            mapping: Column mapping specification (auto-detected if omitted).
            config: Processing configuration parameters (defaults used if omitted).
            dataset_id: Optional dataset identifier.
            dataset_name: Optional human-readable dataset name.

        Returns:
            Tuple of (processed_dataframe, DataProcessingResult).
        """
        start_time = time.perf_counter()
        now = datetime.now(timezone.utc)
        resolved_id = dataset_id or f"proc_{now.strftime('%Y%m%d_%H%M%S')}"
        resolved_name = dataset_name or "Business_Dataset"
        effective_config = config or DataProcessingConfig()
        audit = AuditTrailTracker()

        # 1. Resolve Column Mapping
        effective_mapping = (
            mapping if mapping is not None else auto_detect_column_mapping(list(df.columns))
        )

        input_row_count = len(df)
        input_columns = list(df.columns)
        input_col_count = len(input_columns)

        # 2. Evaluate Baseline Quality Score (Before Processing)
        init_errors, init_warnings = self.validator.validate(df, effective_mapping)
        init_leakage = self.leakage_detector.detect(df, effective_mapping)
        quality_score_before = self.quality_scorer.score(
            df=df,
            mapping=effective_mapping,
            validation_issues=init_errors + init_warnings,
            leakage_report=init_leakage,
        )

        # 3. Step 1: Cleaning & Type Normalization
        current_df = self.cleaner.clean(
            df=df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 4. Step 2: Duplicate Handling
        current_df, dups_before, dups_removed, conflict_count = self.duplicate_handler.resolve_duplicates(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 5. Step 3: Invalid Business Value Rectification
        current_df = self.invalid_handler.rectify_invalid_values(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 6. Step 4: Missing Value Handling
        current_df, missing_before, missing_after = self.missing_handler.handle_missing_values(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 7. Step 5: Chronological Ordering & Gap Integrity
        current_df, ts_integrity = self.ts_analyzer.analyze_and_sort(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 8. Step 6: Outlier Detection & Profiling
        current_df, outlier_summary = self.outlier_detector.detect_and_treat(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 9. Step 7: Leakage-Safe Feature Engineering
        current_df, generated_features = self.feature_engineer.engineer_features(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 10. Step 8: Temporal Splitting Metadata
        _, _, _, split_meta = self.splitter.split(
            df=current_df,
            mapping=effective_mapping,
            config=effective_config,
            audit=audit,
        )

        # 11. Step 9: Re-evaluate Processed Quality Score (After Processing)
        # Note: In processed data, lags at start of series legitimately contain NaNs.
        # To evaluate foundational quality accurately without penalizing engineered lags,
        # we validate the core canonical fields.
        canonical_cols = [
            c for c in [
                effective_mapping.date,
                effective_mapping.entity_id,
                effective_mapping.target,
                effective_mapping.product_id,
                effective_mapping.category,
                effective_mapping.region,
                effective_mapping.store_type,
                effective_mapping.price,
                effective_mapping.promotion,
                effective_mapping.holiday,
                effective_mapping.inventory,
            ]
            if c and c in current_df.columns
        ]
        eval_df = current_df[canonical_cols].copy()
        post_errors, post_warnings = self.validator.validate(eval_df, effective_mapping)
        post_leakage = self.leakage_detector.detect(current_df, effective_mapping)
        quality_score_after = self.quality_scorer.score(
            df=eval_df,
            mapping=effective_mapping,
            validation_issues=post_errors + post_warnings,
            leakage_report=post_leakage,
        )

        # 12. Check Duplicates After
        active_key = [effective_mapping.date, effective_mapping.entity_id]
        if effective_mapping.product_id and effective_mapping.product_id in current_df.columns:
            active_key.append(effective_mapping.product_id)
        valid_key = [k for k in active_key if k in current_df.columns]
        duplicates_after = int(current_df.duplicated(subset=valid_key).sum()) if valid_key else 0

        duration = round(time.perf_counter() - start_time, 4)
        output_row_count = len(current_df)
        output_columns = list(current_df.columns)
        output_col_count = len(output_columns)
        removed_rows = max(0, input_row_count - output_row_count)

        # Status determination
        if len(post_errors) > 0:
            status = "FAILED"
        elif len(post_warnings) > 0 or post_leakage.has_leakage or quality_score_after.overall_score < 80.0:
            status = "WARNING"
        else:
            status = "SUCCESS"

        # Generate sample records (head 5 rows converted to dict)
        sample_records: List[Dict[str, Any]] = []
        for _, row in current_df.head(5).iterrows():
            record_dict = {}
            for col in current_df.columns:
                val = row[col]
                if pd.isna(val):
                    record_dict[col] = None
                elif isinstance(val, (np.integer, int)):
                    record_dict[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    record_dict[col] = round(float(val), 4)
                elif isinstance(val, (pd.Timestamp, datetime)):
                    record_dict[col] = val.isoformat()
                else:
                    record_dict[col] = str(val)
            sample_records.append(record_dict)

        warnings_list: List[str] = [w.message for w in post_warnings]
        if post_leakage.has_leakage:
            warnings_list.extend(post_leakage.warnings)

        result = DataProcessingResult(
            dataset_id=resolved_id,
            dataset_name=resolved_name,
            status=status,
            config=effective_config,
            input_row_count=input_row_count,
            output_row_count=output_row_count,
            input_column_count=input_col_count,
            output_column_count=output_col_count,
            input_columns=input_columns,
            output_columns=output_columns,
            removed_rows_count=removed_rows,
            modified_rows_count=audit.total_rows_affected,
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            duplicates_before=dups_before,
            duplicates_after=duplicates_after,
            conflicting_duplicates_count=conflict_count,
            outlier_summary=outlier_summary,
            time_series_integrity=ts_integrity,
            temporal_split=split_meta,
            generated_features=generated_features,
            quality_score_before=quality_score_before,
            quality_score_after=quality_score_after,
            audit_trail=audit.entries,
            warnings=warnings_list,
            errors=[e.message for e in post_errors],
            processing_duration_seconds=duration,
            processed_sample=sample_records,
            created_at=now,
        )

        return current_df, result
