"""Invalid Business Value Detection and Treatment for GlassBox-BI.

Handles negative target values, non-positive pricing, negative inventory levels,
and non-binary holiday/promotion indicators based on deterministic business rules.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.audit import AuditTrailTracker


class InvalidValueHandler:
    """Validates and rectifies business boundary violations."""

    def rectify_invalid_values(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> pd.DataFrame:
        """Applies configurable business rule corrections.

        Args:
            df: Dataframe after duplicate resolution.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            pd.DataFrame with corrected or clipped business fields.
        """
        rectified = df.copy()

        # 1. Target Validation (Target cannot be negative in retail demand)
        target_col = mapping.target
        if target_col in rectified.columns:
            neg_target_mask = rectified[target_col] < 0
            neg_count = int(neg_target_mask.sum())
            if neg_count > 0:
                rectified.loc[neg_target_mask, target_col] = 0.0
                audit.record(
                    operation="invalid_target_rectification",
                    column=target_col,
                    rows_affected=neg_count,
                    reason="Negative demand values are mathematically impossible in retail context",
                    strategy="clip_to_zero",
                    before_summary={"negative_target_count": neg_count},
                    after_summary={"min_target": float(rectified[target_col].min())},
                )

        # 2. Price Validation (Price must be strictly positive)
        price_col = mapping.price
        if price_col and price_col in rectified.columns:
            non_pos_mask = (rectified[price_col] <= 0) & rectified[price_col].notna()
            non_pos_count = int(non_pos_mask.sum())
            if non_pos_count > 0 and config.clip_negative_prices:
                # Replace <= 0 with positive median price
                valid_prices = rectified.loc[rectified[price_col] > 0, price_col]
                fill_price = float(valid_prices.median()) if len(valid_prices) > 0 else 1.0
                rectified.loc[non_pos_mask, price_col] = fill_price
                audit.record(
                    operation="invalid_price_rectification",
                    column=price_col,
                    rows_affected=non_pos_count,
                    reason="Selling price must be strictly positive",
                    strategy="impute_positive_median",
                    before_summary={"non_positive_price_count": non_pos_count},
                    after_summary={"imputed_price": fill_price},
                )

        # 3. Inventory Validation (Inventory cannot be negative)
        inv_col = mapping.inventory
        if inv_col and inv_col in rectified.columns:
            neg_inv_mask = (rectified[inv_col] < 0) & rectified[inv_col].notna()
            neg_inv_count = int(neg_inv_mask.sum())
            if neg_inv_count > 0 and config.clip_negative_inventory:
                rectified.loc[neg_inv_mask, inv_col] = 0.0
                audit.record(
                    operation="invalid_inventory_rectification",
                    column=inv_col,
                    rows_affected=neg_inv_count,
                    reason="Physical inventory on hand cannot be negative",
                    strategy="clip_to_zero",
                    before_summary={"negative_inventory_count": neg_inv_count},
                    after_summary={"min_inventory": 0.0},
                )

        # 4. Promotion and Holiday Binary Clamping
        for col_name, label in [(mapping.promotion, "promotion"), (mapping.holiday, "holiday")]:
            if col_name and col_name in rectified.columns:
                non_binary_mask = (~rectified[col_name].isin([0, 1])) & rectified[col_name].notna()
                nb_count = int(non_binary_mask.sum())
                if nb_count > 0:
                    # Clip to 0 or 1
                    rectified[col_name] = np.where(rectified[col_name] > 0, 1, 0)
                    audit.record(
                        operation=f"invalid_{label}_indicator_clamping",
                        column=col_name,
                        rows_affected=nb_count,
                        reason=f"{label.capitalize()} indicator must be binary (0 or 1)",
                        strategy="binary_clamping_gt_zero",
                        before_summary={"non_binary_count": nb_count},
                        after_summary={"unique_values": [0, 1]},
                    )

        return rectified
