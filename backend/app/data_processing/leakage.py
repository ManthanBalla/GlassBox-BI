"""Temporal Data Leakage Detection Module for GlassBox-BI.

Inspects tabular datasets for potential temporal lookahead bias, future-derived features,
future timestamps, and recommends strict chronological splitting strategies.
"""

from typing import List, Set
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import TemporalLeakageReport


class TemporalLeakageDetector:
    """Detects suspicious feature patterns that could leak future information into historical records."""

    # Keywords commonly indicating future or lead features
    SUSPICIOUS_FUTURE_TOKENS = {
        "future", "lead", "next", "target_t+", "actuals_t+", "subsequent",
        "following", "post", "tomorrow", "next_day", "next_week", "next_month",
        "forward", "lookahead", "lead_1", "lead_7", "future_sales", "future_target",
    }

    def detect(self, df: pd.DataFrame, mapping: ColumnMapping) -> TemporalLeakageReport:
        """Analyzes dataframe columns and values to flag potential temporal data leakage.

        Args:
            df: Source or mapped tabular dataframe.
            mapping: Active column mapping.

        Returns:
            TemporalLeakageReport with warnings and splitting recommendations.
        """
        flagged_columns: List[str] = []
        warnings: List[str] = []

        # 1. Inspect Column Names for Suspicious Future Tokens
        for col in df.columns:
            col_clean = col.lower().replace("-", "_").replace(" ", "_")
            if any(token in col_clean for token in self.SUSPICIOUS_FUTURE_TOKENS):
                flagged_columns.append(col)
                warnings.append(
                    f"Potential temporal leakage: Column '{col}' matches future lookahead naming conventions ('{col_clean}'). Verify this feature is strictly observable at prediction time."
                )

        # 2. Check for Exact Duplicate Target under a Different Column Name
        target_col = mapping.target
        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors="coerce")
            for col in df.columns:
                if col == target_col or col in (mapping.date, mapping.entity_id):
                    continue
                other_series = pd.to_numeric(df[col], errors="coerce")
                # If numeric and not empty, check correlation / identity
                if not other_series.isna().all() and not target_series.isna().all():
                    if (target_series == other_series).all():
                        flagged_columns.append(col)
                        warnings.append(
                            f"Critical data leakage risk: Column '{col}' is identical to the target column '{target_col}'. Including it as an input will cause severe target leakage."
                        )

        # 3. Check for Date Columns Occurring After the Primary Date Column
        if mapping.date in df.columns:
            try:
                primary_dates = pd.to_datetime(df[mapping.date], errors="coerce")
                for col in df.columns:
                    if col == mapping.date or col in flagged_columns:
                        continue
                    # Check if column is datetime-like
                    if "date" in col.lower() or "time" in col.lower() or df[col].dtype.kind == "M":
                        parsed_col = pd.to_datetime(df[col], errors="coerce")
                        if parsed_col.notna().sum() > 0.5 * len(df):
                            # Compare dates where both are valid
                            valid_mask = primary_dates.notna() & parsed_col.notna()
                            if (parsed_col[valid_mask] > primary_dates[valid_mask]).any():
                                flagged_columns.append(col)
                                warnings.append(
                                    f"Potential temporal leakage: Date column '{col}' contains timestamps occurring after the primary observation date '{mapping.date}'."
                                )
            except Exception:
                pass  # Graceful fallback if date parsing fails

        has_leakage = len(flagged_columns) > 0
        deduped_flags = list(dict.fromkeys(flagged_columns))

        recommendation = (
            "Use chronological walk-forward train/validation/test splitting for forecasting. "
            "Never use random k-fold cross-validation or shuffle time-series observations, "
            "and ensure all features are strictly lagging or observable at decision time."
        )

        return TemporalLeakageReport(
            has_leakage=has_leakage,
            flagged_columns=deduped_flags,
            warnings=warnings,
            splitting_recommendation=recommendation,
        )
