"""Missing Value Handling and Imputation for GlassBox-BI.

Applies deterministic, temporal-safe missing value imputation.
Zero Leakage Rule: Target values are NEVER filled from future observations.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import (
    DataProcessingConfig,
    MissingCategoricalStrategy,
    MissingNumericStrategy,
    MissingTargetStrategy,
)
from backend.app.data_processing.processing.audit import AuditTrailTracker


class MissingValueHandler:
    """Handles missing values across numeric, categorical, and target dimensions."""

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """Applies missing value strategies and returns before/after missingness dictionaries.

        Args:
            df: Dataframe after duplicate and invalid value rectification.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Tuple of (imputed_df, missing_before_dict, missing_after_dict).
        """
        missing_before = {col: int(df[col].isna().sum()) for col in df.columns}
        processed = df.copy()

        # 1. Target Missing Value Handling (Strict Zero-Leakage Policy)
        target_col = mapping.target
        if target_col in processed.columns:
            target_nulls = int(processed[target_col].isna().sum())
            if target_nulls > 0:
                if config.missing_target_strategy == MissingTargetStrategy.DROP:
                    initial_len = len(processed)
                    processed = processed.dropna(subset=[target_col]).copy()
                    dropped_count = initial_len - len(processed)
                    audit.record(
                        operation="missing_target_drop",
                        column=target_col,
                        rows_affected=dropped_count,
                        reason="Zero-leakage policy: Missing historical targets cannot be fabricated from future data",
                        strategy="drop_na",
                        before_summary={"missing_target_count": target_nulls},
                        after_summary={"retained_rows": len(processed)},
                    )
                elif config.missing_target_strategy == MissingTargetStrategy.FLAG:
                    processed["missing_target_flag"] = processed[target_col].isna().astype(int)
                    audit.record(
                        operation="missing_target_flag",
                        column=target_col,
                        rows_affected=target_nulls,
                        reason="Mark observations with missing target without synthetic imputation",
                        strategy="flag_binary",
                    )

        # 2. Numeric Features Imputation
        numeric_cols = [
            mapping.price,
            mapping.inventory,
        ]
        active_numeric = [c for c in numeric_cols if c and c in processed.columns]

        entity_col = mapping.entity_id if mapping.entity_id in processed.columns else None

        for col in active_numeric:
            col_nulls = int(processed[col].isna().sum())
            if col_nulls == 0:
                continue

            if config.missing_numeric_strategy == MissingNumericStrategy.MEDIAN:
                # Group-aware median if entity exists, fallback to global median
                if entity_col:
                    group_medians = processed.groupby(entity_col)[col].transform("median")
                    processed[col] = processed[col].fillna(group_medians)
                global_median = processed[col].median()
                fill_val = float(global_median) if pd.notna(global_median) else config.numeric_constant_fallback
                processed[col] = processed[col].fillna(fill_val)

                audit.record(
                    operation="missing_numeric_imputation",
                    column=col,
                    rows_affected=col_nulls,
                    reason=f"Impute missing numeric values in {col}",
                    strategy="entity_grouped_median_with_global_fallback",
                    before_summary={"null_count": col_nulls},
                    after_summary={"remaining_nulls": int(processed[col].isna().sum())},
                )

            elif config.missing_numeric_strategy == MissingNumericStrategy.FORWARD_FILL:
                # Group-aware forward fill (temporally safe; uses previous known state)
                if entity_col:
                    processed[col] = processed.groupby(entity_col)[col].ffill()
                else:
                    processed[col] = processed[col].ffill()
                # Constant fallback for initial observations with no prior history
                processed[col] = processed[col].fillna(config.numeric_constant_fallback)

                audit.record(
                    operation="missing_numeric_imputation",
                    column=col,
                    rows_affected=col_nulls,
                    reason=f"Temporal forward-fill imputation for {col}",
                    strategy="forward_fill_with_constant_fallback",
                    before_summary={"null_count": col_nulls},
                    after_summary={"remaining_nulls": int(processed[col].isna().sum())},
                )

            elif config.missing_numeric_strategy == MissingNumericStrategy.CONSTANT:
                processed[col] = processed[col].fillna(config.numeric_constant_fallback)
                audit.record(
                    operation="missing_numeric_imputation",
                    column=col,
                    rows_affected=col_nulls,
                    reason=f"Constant replacement for {col}",
                    strategy=f"constant_{config.numeric_constant_fallback}",
                    before_summary={"null_count": col_nulls},
                    after_summary={"remaining_nulls": int(processed[col].isna().sum())},
                )

        # 3. Categorical Dimensions Imputation
        cat_cols = [
            mapping.category,
            mapping.region,
            mapping.store_type,
            mapping.product_id,
        ]
        active_cat = [c for c in cat_cols if c and c in processed.columns]

        for col in active_cat:
            col_nulls = int(processed[col].isna().sum())
            if col_nulls == 0:
                continue

            if config.missing_categorical_strategy == MissingCategoricalStrategy.UNKNOWN:
                processed[col] = processed[col].fillna("Unknown")
                audit.record(
                    operation="missing_categorical_imputation",
                    column=col,
                    rows_affected=col_nulls,
                    reason=f"Explicit unknown categorization for missing {col}",
                    strategy="unknown_category_token",
                    before_summary={"null_count": col_nulls},
                    after_summary={"remaining_nulls": 0},
                )
            elif config.missing_categorical_strategy == MissingCategoricalStrategy.MODE:
                mode_series = processed[col].mode()
                fill_mode = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
                processed[col] = processed[col].fillna(fill_mode)
                audit.record(
                    operation="missing_categorical_imputation",
                    column=col,
                    rows_affected=col_nulls,
                    reason=f"Mode imputation for missing {col}",
                    strategy=f"mode_value_{fill_mode}",
                    before_summary={"null_count": col_nulls},
                    after_summary={"remaining_nulls": 0},
                )

        # 4. Binary Indicator Imputation (Default to 0 / False if missing)
        for ind_col in [mapping.promotion, mapping.holiday]:
            if ind_col and ind_col in processed.columns:
                ind_nulls = int(processed[ind_col].isna().sum())
                if ind_nulls > 0:
                    processed[ind_col] = processed[ind_col].fillna(0).astype(int)
                    audit.record(
                        operation="missing_indicator_imputation",
                        column=ind_col,
                        rows_affected=ind_nulls,
                        reason=f"Missing indicator flag {ind_col} safely assumed 0 (non-active)",
                        strategy="zero_imputation",
                        before_summary={"null_count": ind_nulls},
                        after_summary={"remaining_nulls": 0},
                    )

        missing_after = {col: int(processed[col].isna().sum()) for col in processed.columns}
        return processed, missing_before, missing_after
