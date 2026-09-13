"""Data Cleaning and Type Normalization for GlassBox-BI.

Ensures incoming canonical data conforms strictly to expected numeric and
categorical representations, handles empty strings, strips whitespace, and
standardizes dates without modifying underlying business signals.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.audit import AuditTrailTracker


class DataCleaner:
    """Normalizes types, strings, and date representations for canonical datasets."""

    def clean(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> pd.DataFrame:
        """Executes non-destructive data cleaning and type normalization.

        Args:
            df: Input dataframe (canonical representation).
            mapping: Column mapping indicating canonical roles.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Cleaned pd.DataFrame with normalized types and whitespace.
        """
        cleaned = df.copy()

        # 1. Normalize String / Object Columns: Strip Whitespace & Replace Empty with NaN
        string_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in string_cols:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(["", "nan", "None", "null", "NULL", "N/A"], np.nan)

        # 2. Standardize Date Column
        date_col = mapping.date
        if date_col in cleaned.columns:
            orig_nulls = cleaned[date_col].isna().sum()
            cleaned[date_col] = pd.to_datetime(cleaned[date_col], errors="coerce")
            new_nulls = cleaned[date_col].isna().sum()
            coerced_count = int(new_nulls - orig_nulls)
            
            audit.record(
                operation="date_standardization",
                column=date_col,
                rows_affected=len(cleaned),
                reason="Standardize dates to datetime64[ns] ISO representation",
                strategy="pd.to_datetime with coerce",
                after_summary={"unparseable_dates_coerced": max(0, coerced_count)},
            )

        # 3. Standardize Target Column (Float)
        target_col = mapping.target
        if target_col in cleaned.columns:
            cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")

        # 4. Standardize Optional Numeric Columns
        numeric_fields = [
            (mapping.price, "float64"),
            (mapping.inventory, "float64"),
        ]
        for col_name, dtype in numeric_fields:
            if col_name and col_name in cleaned.columns:
                cleaned[col_name] = pd.to_numeric(cleaned[col_name], errors="coerce")

        # 5. Standardize Promotion and Holiday Flags (Categorical / Binary Int)
        indicator_fields = [mapping.promotion, mapping.holiday]
        for col_name in indicator_fields:
            if col_name and col_name in cleaned.columns:
                # Coerce to numeric first
                cleaned[col_name] = pd.to_numeric(cleaned[col_name], errors="coerce")

        audit.record(
            operation="type_normalization",
            reason="Coerce numeric and indicator fields to standard floating and binary representations",
            strategy="pd.to_numeric with coercion",
            rows_affected=len(cleaned),
            after_summary={"dtypes": {c: str(cleaned[c].dtype) for c in cleaned.columns}},
        )

        return cleaned
