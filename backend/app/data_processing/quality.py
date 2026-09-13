"""Transparent, Deterministic Data Quality Scoring Engine for GlassBox-BI.

Calculates an explainable 0–100 data quality score across 6 explicit dimensions:
Schema, Missing Data, Duplicates, Temporal Integrity, Numeric Validity, and Consistency.
NO opaque ML models — strictly arithmetic, auditable, and rule-based.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import (
    QualityScoreBreakdown,
    TemporalLeakageReport,
    ValidationIssue,
    ValidationSeverity,
)


class DataQualityScorer:
    """Computes a deterministic, explainable data quality score."""

    DEFAULT_WEIGHTS = {
        "schema": 0.20,
        "missing_data": 0.20,
        "duplicate": 0.15,
        "temporal_integrity": 0.20,
        "numeric_validity": 0.15,
        "consistency": 0.10,
    }

    def score(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        validation_issues: List[ValidationIssue],
        leakage_report: TemporalLeakageReport,
    ) -> QualityScoreBreakdown:
        """Calculates multi-dimensional data quality scores with transparent deduction logs."""
        deductions: List[Dict[str, Any]] = []
        total_rows = len(df) if len(df) > 0 else 1
        total_cells = df.size if df.size > 0 else 1

        # 1. Schema Quality (0-100)
        schema_score = 100.0
        missing_req = [i for i in validation_issues if i.issue_type == "MISSING_REQUIRED_COLUMN"]
        if missing_req:
            penalty = min(80.0, len(missing_req) * 35.0)
            schema_score -= penalty
            deductions.append({
                "dimension": "Schema",
                "penalty": penalty,
                "reason": f"Missing {len(missing_req)} required column(s): {', '.join([i.field or '' for i in missing_req])}",
            })

        dupe_cols = [i for i in validation_issues if i.issue_type == "DUPLICATE_COLUMN_NAMES"]
        if dupe_cols:
            schema_score -= 20.0
            deductions.append({
                "dimension": "Schema",
                "penalty": 20.0,
                "reason": "Duplicate column names detected in source dataset.",
            })

        schema_score = max(0.0, min(100.0, schema_score))

        # 2. Missing Data Quality (0-100)
        missing_data_score = 100.0
        total_nulls = int(df.isna().sum().sum())
        null_rate = total_nulls / total_cells

        # Global null deduction: 1 point per 1% nulls
        global_penalty = min(50.0, null_rate * 100.0)
        if global_penalty > 0:
            missing_data_score -= global_penalty
            deductions.append({
                "dimension": "Missing Data",
                "penalty": round(global_penalty, 2),
                "reason": f"Global missingness rate is {null_rate:.2%} ({total_nulls} null cells).",
            })

        # Required column null penalty
        req_nulls = [i for i in validation_issues if i.issue_type == "NULL_IN_REQUIRED_COLUMN"]
        for r_null in req_nulls:
            col_penalty = min(25.0, (r_null.details or {}).get("null_pct", 0.1) * 100.0 * 2)
            missing_data_score -= col_penalty
            deductions.append({
                "dimension": "Missing Data",
                "penalty": round(col_penalty, 2),
                "reason": f"Required column '{r_null.field}' contains missing values.",
            })

        missing_data_score = max(0.0, min(100.0, missing_data_score))

        # 3. Duplicate Quality (0-100)
        duplicate_score = 100.0
        exact_dupes = int(df.duplicated().sum())
        if exact_dupes > 0:
            dupe_rate = exact_dupes / total_rows
            penalty = min(50.0, dupe_rate * 100.0 * 2)
            duplicate_score -= penalty
            deductions.append({
                "dimension": "Duplicates",
                "penalty": round(penalty, 2),
                "reason": f"Found {exact_dupes} exact duplicate rows ({dupe_rate:.2%}).",
            })

        key_cols = [c for c in [mapping.date, mapping.entity_id, mapping.product_id] if c and c in df.columns]
        if len(key_cols) >= 2:
            key_dupes = int(df.duplicated(subset=key_cols).sum())
            if key_dupes > 0:
                key_rate = key_dupes / total_rows
                penalty = min(30.0, key_rate * 100.0)
                duplicate_score -= penalty
                deductions.append({
                    "dimension": "Duplicates",
                    "penalty": round(penalty, 2),
                    "reason": f"Found {key_dupes} duplicate time-series keys ({', '.join(key_cols)}).",
                })

        duplicate_score = max(0.0, min(100.0, duplicate_score))

        # 4. Temporal Integrity (0-100)
        temporal_score = 100.0
        unparseable = [i for i in validation_issues if i.issue_type == "UNPARSEABLE_DATES"]
        if unparseable:
            temporal_score -= 50.0
            deductions.append({
                "dimension": "Temporal Integrity",
                "penalty": 50.0,
                "reason": "Date column contains invalid or unparseable timestamps.",
            })

        monotonic_warn = [i for i in validation_issues if i.issue_type == "NON_MONOTONIC_DATES"]
        if monotonic_warn:
            temporal_score -= 5.0
            deductions.append({
                "dimension": "Temporal Integrity",
                "penalty": 5.0,
                "reason": "Timestamps are not ordered chronologically.",
            })

        if leakage_report.has_leakage:
            leakage_penalty = min(30.0, len(leakage_report.flagged_columns) * 15.0)
            temporal_score -= leakage_penalty
            deductions.append({
                "dimension": "Temporal Integrity",
                "penalty": leakage_penalty,
                "reason": f"Potential temporal data leakage flagged in columns: {', '.join(leakage_report.flagged_columns)}.",
            })

        temporal_score = max(0.0, min(100.0, temporal_score))

        # 5. Numeric Validity (0-100)
        numeric_score = 100.0
        neg_targets = [i for i in validation_issues if i.issue_type == "NEGATIVE_TARGET_VALUES"]
        if neg_targets:
            penalty = 40.0
            numeric_score -= penalty
            deductions.append({
                "dimension": "Numeric Validity",
                "penalty": penalty,
                "reason": "Target variable contains negative values, which is invalid for demand forecasting.",
            })

        non_num_target = [i for i in validation_issues if i.issue_type == "NON_NUMERIC_TARGET"]
        if non_num_target:
            numeric_score -= 40.0
            deductions.append({
                "dimension": "Numeric Validity",
                "penalty": 40.0,
                "reason": "Target column contains non-numeric entries.",
            })

        neg_inv = [i for i in validation_issues if i.issue_type == "NEGATIVE_INVENTORY"]
        if neg_inv:
            numeric_score -= 10.0
            deductions.append({
                "dimension": "Numeric Validity",
                "penalty": 10.0,
                "reason": "Inventory column contains negative values.",
            })

        non_pos_price = [i for i in validation_issues if i.issue_type == "NON_POSITIVE_PRICE"]
        if non_pos_price:
            numeric_score -= 10.0
            deductions.append({
                "dimension": "Numeric Validity",
                "penalty": 10.0,
                "reason": "Price column contains zero or negative values.",
            })

        numeric_score = max(0.0, min(100.0, numeric_score))

        # 6. Consistency Quality (0-100)
        consistency_score = 100.0
        short_series = [i for i in validation_issues if i.issue_type == "SHORT_TIME_SERIES"]
        if short_series:
            consistency_score -= 15.0
            deductions.append({
                "dimension": "Consistency",
                "penalty": 15.0,
                "reason": "Some business entities have too few observations for reliable time-series modeling.",
            })

        non_binary = [i for i in validation_issues if i.issue_type == "NON_BINARY_FLAG"]
        if non_binary:
            consistency_score -= 5.0
            deductions.append({
                "dimension": "Consistency",
                "penalty": 5.0,
                "reason": "Indicator flags (promotion/holiday) contain non-binary values.",
            })

        consistency_score = max(0.0, min(100.0, consistency_score))

        # Overall Weighted Composite Score
        overall = (
            self.DEFAULT_WEIGHTS["schema"] * schema_score
            + self.DEFAULT_WEIGHTS["missing_data"] * missing_data_score
            + self.DEFAULT_WEIGHTS["duplicate"] * duplicate_score
            + self.DEFAULT_WEIGHTS["temporal_integrity"] * temporal_score
            + self.DEFAULT_WEIGHTS["numeric_validity"] * numeric_score
            + self.DEFAULT_WEIGHTS["consistency"] * consistency_score
        )
        overall = round(max(0.0, min(100.0, overall)), 1)

        # Grade Mapping
        if overall >= 95.0:
            grade = "A+"
        elif overall >= 90.0:
            grade = "A"
        elif overall >= 80.0:
            grade = "B"
        elif overall >= 70.0:
            grade = "C"
        elif overall >= 60.0:
            grade = "D"
        else:
            grade = "F"

        return QualityScoreBreakdown(
            schema_score=round(schema_score, 1),
            missing_data_score=round(missing_data_score, 1),
            duplicate_score=round(duplicate_score, 1),
            temporal_integrity_score=round(temporal_score, 1),
            numeric_validity_score=round(numeric_score, 1),
            consistency_score=round(consistency_score, 1),
            overall_score=overall,
            grade=grade,
            deductions=deductions,
        )
