"""Comprehensive Data Validation Engine for GlassBox-BI.

Validates tabular time-series datasets against the canonical business data contract,
evaluating schema integrity, null rates, duplicates, temporal continuity, and numeric bounds.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import ValidationIssue, ValidationSeverity


class DataValidator:
    """Performs multi-dimensional data validation on tabular business time-series."""

    def __init__(
        self,
        missing_threshold_warn: float = 0.05,
        missing_threshold_err: float = 0.40,
        min_series_length: int = 10,
    ):
        self.missing_threshold_warn = missing_threshold_warn
        self.missing_threshold_err = missing_threshold_err
        self.min_series_length = min_series_length

    def validate(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
    ) -> Tuple[List[ValidationIssue], List[ValidationIssue]]:
        """Executes full validation suite on dataframe using provided column mapping.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        # A. Schema Validation
        self._validate_schema(df, mapping, errors, warnings)

        # If critical required columns are completely missing, return early
        critical_missing = any(
            err.issue_type == "MISSING_REQUIRED_COLUMN" for err in errors
        )
        if critical_missing:
            return errors, warnings

        # B. Missing Values Validation
        self._validate_missing_values(df, mapping, errors, warnings)

        # C. Duplicate Records Validation
        self._validate_duplicates(df, mapping, errors, warnings)

        # D. Date & Temporal Validation
        self._validate_dates(df, mapping, errors, warnings)

        # E. Numeric Validation
        self._validate_numeric(df, mapping, errors, warnings)

        # F. Time-Series Integrity & Series Length
        self._validate_time_series_integrity(df, mapping, errors, warnings)

        return errors, warnings

    def _validate_schema(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Validates column presence and uniqueness."""
        # Check duplicate column names in source
        if len(df.columns) != len(set(df.columns)):
            errors.append(
                ValidationIssue(
                    issue_type="DUPLICATE_COLUMN_NAMES",
                    message="The dataset contains duplicate column names.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Required columns in source
        required_fields = [
            ("date", mapping.date),
            ("entity_id", mapping.entity_id),
            ("target", mapping.target),
        ]

        for field_name, source_col in required_fields:
            if not source_col or source_col not in df.columns:
                errors.append(
                    ValidationIssue(
                        field=source_col or field_name,
                        issue_type="MISSING_REQUIRED_COLUMN",
                        message=f"Required column '{source_col or field_name}' for canonical '{field_name}' not found in dataset.",
                        severity=ValidationSeverity.ERROR,
                        details={"canonical_field": field_name, "mapped_to": source_col},
                    )
                )

        # Optional mapped columns that are missing from source
        optional_fields = [
            ("product_id", mapping.product_id),
            ("category", mapping.category),
            ("region", mapping.region),
            ("store_type", mapping.store_type),
            ("price", mapping.price),
            ("promotion", mapping.promotion),
            ("holiday", mapping.holiday),
            ("inventory", mapping.inventory),
        ]
        for field_name, source_col in optional_fields:
            if source_col and source_col not in df.columns:
                warnings.append(
                    ValidationIssue(
                        field=source_col,
                        issue_type="MISSING_OPTIONAL_COLUMN",
                        message=f"Optional mapped column '{source_col}' (for '{field_name}') was not found in dataset.",
                        severity=ValidationSeverity.WARNING,
                    )
                )

    def _validate_missing_values(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Inspects null rates across required and optional columns."""
        total_rows = len(df)
        if total_rows == 0:
            errors.append(
                ValidationIssue(
                    issue_type="EMPTY_DATASET",
                    message="Dataset contains 0 rows.",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        # Check required columns nulls
        for col_name in [mapping.date, mapping.entity_id, mapping.target]:
            if col_name in df.columns:
                null_count = int(df[col_name].isna().sum())
                if null_count > 0:
                    errors.append(
                        ValidationIssue(
                            field=col_name,
                            issue_type="NULL_IN_REQUIRED_COLUMN",
                            message=f"Required column '{col_name}' contains {null_count} null/missing values.",
                            severity=ValidationSeverity.ERROR,
                            details={"null_count": null_count, "null_pct": round(null_count / total_rows, 4)},
                        )
                    )

        # Check all other columns for excessive missingness
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = null_count / total_rows
            if null_pct >= self.missing_threshold_err:
                errors.append(
                    ValidationIssue(
                        field=col,
                        issue_type="EXCESSIVE_MISSING_DATA",
                        message=f"Column '{col}' has {null_pct:.1%} missing data, exceeding critical threshold ({self.missing_threshold_err:.1%}).",
                        severity=ValidationSeverity.ERROR,
                        details={"null_pct": round(null_pct, 4)},
                    )
                )
            elif null_pct >= self.missing_threshold_warn:
                warnings.append(
                    ValidationIssue(
                        field=col,
                        issue_type="HIGH_MISSING_DATA",
                        message=f"Column '{col}' has {null_pct:.1%} missing data, exceeding warning threshold ({self.missing_threshold_warn:.1%}).",
                        severity=ValidationSeverity.WARNING,
                        details={"null_pct": round(null_pct, 4)},
                    )
                )

    def _validate_duplicates(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Inspects full row duplicates and entity/product/date primary key duplicates."""
        exact_dupes = int(df.duplicated().sum())
        if exact_dupes > 0:
            warnings.append(
                ValidationIssue(
                    issue_type="EXACT_DUPLICATE_ROWS",
                    message=f"Found {exact_dupes} exact duplicate rows.",
                    severity=ValidationSeverity.WARNING,
                    details={"duplicate_count": exact_dupes},
                )
            )

        # Check composite business key duplicates (date + entity_id [+ product_id])
        key_cols = [c for c in [mapping.date, mapping.entity_id, mapping.product_id] if c and c in df.columns]
        if len(key_cols) >= 2:
            key_dupes = int(df.duplicated(subset=key_cols).sum())
            if key_dupes > 0:
                warnings.append(
                    ValidationIssue(
                        field=",".join(key_cols),
                        issue_type="DUPLICATE_TIME_SERIES_KEYS",
                        message=f"Found {key_dupes} records with duplicate key combinations ({', '.join(key_cols)}). Multiple observations exist for the same timestamp.",
                        severity=ValidationSeverity.WARNING,
                        details={"duplicate_key_count": key_dupes},
                    )
                )

    def _validate_dates(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Validates date parseability and chronological ordering."""
        date_col = mapping.date
        if date_col not in df.columns:
            return

        date_series = pd.to_datetime(df[date_col], errors="coerce")
        unparseable = int(date_series.isna().sum())
        if unparseable > 0:
            errors.append(
                ValidationIssue(
                    field=date_col,
                    issue_type="UNPARSEABLE_DATES",
                    message=f"Date column '{date_col}' contains {unparseable} invalid or unparseable timestamps.",
                    severity=ValidationSeverity.ERROR,
                    details={"unparseable_count": unparseable},
                )
            )
            return

        # Check if dates are generally chronological
        if not date_series.is_monotonic_increasing:
            warnings.append(
                ValidationIssue(
                    field=date_col,
                    issue_type="NON_MONOTONIC_DATES",
                    message=f"Date column '{date_col}' is not sorted chronologically. Automatic sorting will be applied.",
                    severity=ValidationSeverity.INFO,
                )
            )

    def _validate_numeric(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Validates numeric ranges, checking for negative demand, prices, or inventory."""
        target_col = mapping.target
        if target_col in df.columns:
            numeric_target = pd.to_numeric(df[target_col], errors="coerce")
            nan_count = int(numeric_target.isna().sum())
            if nan_count > 0:
                errors.append(
                    ValidationIssue(
                        field=target_col,
                        issue_type="NON_NUMERIC_TARGET",
                        message=f"Target column '{target_col}' contains {nan_count} non-numeric values.",
                        severity=ValidationSeverity.ERROR,
                    )
                )
            else:
                neg_count = int((numeric_target < 0).sum())
                if neg_count > 0:
                    errors.append(
                        ValidationIssue(
                            field=target_col,
                            issue_type="NEGATIVE_TARGET_VALUES",
                            message=f"Target column '{target_col}' contains {neg_count} negative values.",
                            severity=ValidationSeverity.ERROR,
                            details={"negative_count": neg_count, "min_value": float(numeric_target.min())},
                        )
                    )

        # Validate Price
        if mapping.price and mapping.price in df.columns:
            prices = pd.to_numeric(df[mapping.price], errors="coerce").dropna()
            if (prices <= 0).any():
                warnings.append(
                    ValidationIssue(
                        field=mapping.price,
                        issue_type="NON_POSITIVE_PRICE",
                        message=f"Price column '{mapping.price}' contains non-positive values.",
                        severity=ValidationSeverity.WARNING,
                    )
                )

        # Validate Inventory
        if mapping.inventory and mapping.inventory in df.columns:
            inv = pd.to_numeric(df[mapping.inventory], errors="coerce").dropna()
            if (inv < 0).any():
                warnings.append(
                    ValidationIssue(
                        field=mapping.inventory,
                        issue_type="NEGATIVE_INVENTORY",
                        message=f"Inventory column '{mapping.inventory}' contains negative stock values.",
                        severity=ValidationSeverity.WARNING,
                    )
                )

        # Validate binary indicator flags (promotion, holiday)
        for flag_field, flag_col in [("promotion", mapping.promotion), ("holiday", mapping.holiday)]:
            if flag_col and flag_col in df.columns:
                unique_vals = set(df[flag_col].dropna().unique())
                # If numeric, should ideally be {0, 1}
                if not unique_vals.issubset({0, 1, 0.0, 1.0, "0", "1", True, False}):
                    warnings.append(
                        ValidationIssue(
                            field=flag_col,
                            issue_type="NON_BINARY_FLAG",
                            message=f"Flag column '{flag_col}' contains non-binary values ({list(unique_vals)[:5]}).",
                            severity=ValidationSeverity.INFO,
                        )
                    )

    def _validate_time_series_integrity(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        """Inspects series lengths per entity to detect sparse or short series."""
        if mapping.entity_id not in df.columns:
            return

        counts = df[mapping.entity_id].value_counts()
        short_series = counts[counts < self.min_series_length]
        if len(short_series) > 0:
            warnings.append(
                ValidationIssue(
                    field=mapping.entity_id,
                    issue_type="SHORT_TIME_SERIES",
                    message=f"{len(short_series)} entities have fewer than {self.min_series_length} observations.",
                    severity=ValidationSeverity.WARNING,
                    details={"short_entity_count": len(short_series)},
                )
            )
