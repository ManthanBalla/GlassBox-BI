"""Outlier Detection and Treatment for GlassBox-BI.

Detects statistical demand anomalies and extreme values.
CRITICAL PRINCIPLE: Spikes caused by promotions, holidays, or business events
represent genuine demand signals. Default behavior is DETECT & FLAG rather than DELETE.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import (
    DataProcessingConfig,
    OutlierMethod,
    OutlierStrategy,
    OutlierSummary,
)
from backend.app.data_processing.processing.audit import AuditTrailTracker


class OutlierDetector:
    """Detects demand and feature outliers using IQR or Z-score thresholds."""

    def detect_and_treat(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> Tuple[pd.DataFrame, OutlierSummary]:
        """Detects outliers and applies the configured treatment strategy.

        Args:
            df: Chronologically sorted dataframe.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Tuple of (treated_df, OutlierSummary).
        """
        treated = df.copy()
        target_col = mapping.target

        if target_col not in treated.columns or len(treated) == 0:
            summary = OutlierSummary(
                method=config.outlier_method.value,
                strategy=config.outlier_strategy.value,
                total_outliers=0,
                outlier_percentage=0.0,
                affected_entities_count=0,
            )
            return treated, summary

        # 1. Compute Outlier Mask
        values = treated[target_col].to_numpy()
        outlier_mask = np.zeros(len(treated), dtype=bool)
        lower_bound: float = 0.0
        upper_bound: float = 0.0

        if config.outlier_method == OutlierMethod.IQR:
            q25 = float(np.nanpercentile(values, 25))
            q75 = float(np.nanpercentile(values, 75))
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - config.outlier_threshold * iqr)
            upper_bound = q75 + config.outlier_threshold * iqr
            outlier_mask = (values < lower_bound) | (values > upper_bound)
        elif config.outlier_method == OutlierMethod.ZSCORE:
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
            if std_val > 1e-6:
                z_scores = np.abs((values - mean_val) / std_val)
                outlier_mask = z_scores > config.outlier_threshold
                lower_bound = max(0.0, mean_val - config.outlier_threshold * std_val)
                upper_bound = mean_val + config.outlier_threshold * std_val
            else:
                outlier_mask = np.zeros(len(treated), dtype=bool)

        total_outliers = int(np.sum(outlier_mask))
        outlier_pct = round((total_outliers / max(1, len(treated))) * 100.0, 2)

        # Count affected entities
        affected_entities = 0
        if mapping.entity_id in treated.columns and total_outliers > 0:
            affected_entities = int(treated.loc[outlier_mask, mapping.entity_id].nunique())

        # 2. Apply Treatment Strategy (Default: FLAG)
        if config.outlier_strategy == OutlierStrategy.FLAG:
            treated["target_outlier"] = outlier_mask.astype(int)
            audit.record(
                operation="outlier_detection_flagging",
                column=target_col,
                rows_affected=total_outliers,
                reason="Flag statistical demand anomalies without destroying genuine promotional/holiday spikes",
                strategy=f"{config.outlier_method.value}_threshold_{config.outlier_threshold}",
                before_summary={"total_outliers_flagged": total_outliers, "pct": outlier_pct},
                after_summary={"flag_column": "target_outlier"},
            )

        elif config.outlier_strategy in (OutlierStrategy.CLIP, OutlierStrategy.WINSORIZE):
            treated["target_original"] = treated[target_col].copy()
            treated.loc[treated[target_col] < lower_bound, target_col] = lower_bound
            treated.loc[treated[target_col] > upper_bound, target_col] = upper_bound
            treated["target_outlier"] = outlier_mask.astype(int)
            audit.record(
                operation="outlier_clipping",
                column=target_col,
                rows_affected=total_outliers,
                reason="Clip statistical extremes to boundary thresholds",
                strategy="clip_to_bounds",
                before_summary={"total_outliers": total_outliers},
                after_summary={"lower_bound": lower_bound, "upper_bound": upper_bound},
            )

        elif config.outlier_strategy == OutlierStrategy.REMOVE:
            initial_count = len(treated)
            treated = treated[~outlier_mask].copy()
            audit.record(
                operation="outlier_removal",
                column=target_col,
                rows_affected=total_outliers,
                reason="Remove statistical outlier rows",
                strategy="drop_outlier_rows",
                before_summary={"initial_rows": initial_count},
                after_summary={"retained_rows": len(treated)},
            )

        summary = OutlierSummary(
            method=config.outlier_method.value,
            strategy=config.outlier_strategy.value,
            total_outliers=total_outliers,
            outlier_percentage=outlier_pct,
            affected_entities_count=affected_entities,
            lower_bound=round(lower_bound, 3),
            upper_bound=round(upper_bound, 3),
        )

        return treated, summary
