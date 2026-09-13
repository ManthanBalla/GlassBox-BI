"""Generic Forecasting and Business Feature Engineering for GlassBox-BI.

Constructs strictly historical calendar, lag, rolling, and business regressors.
ZERO LEAKAGE PRINCIPLE: All lag and rolling calculations use past data exclusively.
Rolling features strictly apply shift(1) before computing rolling windows.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.audit import AuditTrailTracker


class FeatureEngineer:
    """Generates leakage-safe temporal and business features for forecasting."""

    def engineer_features(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Constructs calendar, lag, rolling, and business features.

        Args:
            df: Chronologically sorted dataframe.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Tuple of (feature_df, list_of_generated_feature_names).
        """
        if not config.feature_engineering_enabled:
            return df, []

        fe_df = df.copy()
        generated_cols: List[str] = []

        date_col = mapping.date
        target_col = mapping.target
        entity_col = mapping.entity_id
        product_col = mapping.product_id if mapping.product_id and mapping.product_id in fe_df.columns else None

        # Determine Grouping Keys for Time-Series Hierarchy
        group_keys = [entity_col]
        if product_col:
            group_keys.append(product_col)
        active_group_keys = [k for k in group_keys if k in fe_df.columns]

        # 1. Calendar Features (Derived strictly from observation timestamp)
        if config.calendar_features_enabled and date_col in fe_df.columns:
            dates = pd.to_datetime(fe_df[date_col])
            
            fe_df["year"] = dates.dt.year.astype(int)
            fe_df["month"] = dates.dt.month.astype(int)
            fe_df["quarter"] = dates.dt.quarter.astype(int)
            fe_df["day_of_week"] = dates.dt.dayofweek.astype(int)
            fe_df["day_of_month"] = dates.dt.day.astype(int)
            fe_df["day_of_year"] = dates.dt.dayofyear.astype(int)
            fe_df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
            fe_df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

            cal_cols = [
                "year", "month", "quarter", "day_of_week",
                "day_of_month", "day_of_year", "week_of_year", "is_weekend"
            ]
            generated_cols.extend(cal_cols)

            audit.record(
                operation="calendar_feature_engineering",
                reason="Generate deterministic temporal attributes from observation date",
                strategy="datetime_component_extraction",
                rows_affected=len(fe_df),
                after_summary={"calendar_features": cal_cols},
            )

        # 2. Historical Lag Features (Strictly past target values)
        if config.lag_features_enabled and target_col in fe_df.columns:
            lag_cols_added: List[str] = []
            
            if active_group_keys:
                grouped_target = fe_df.groupby(active_group_keys)[target_col]
                for lag in config.lags:
                    col_name = f"lag_{lag}"
                    fe_df[col_name] = grouped_target.shift(lag)
                    lag_cols_added.append(col_name)
            else:
                for lag in config.lags:
                    col_name = f"lag_{lag}"
                    fe_df[col_name] = fe_df[target_col].shift(lag)
                    lag_cols_added.append(col_name)

            generated_cols.extend(lag_cols_added)
            audit.record(
                operation="lag_feature_engineering",
                column=target_col,
                reason="Generate strictly historical target lags (no future lookahead)",
                strategy=f"grouped_shift_lags_{config.lags}",
                rows_affected=len(fe_df),
                after_summary={"lag_features": lag_cols_added},
            )

        # 3. Rolling Window Features (Shift-1 Before Rolling = Zero Leakage)
        if config.rolling_features_enabled and target_col in fe_df.columns:
            roll_cols_added: List[str] = []

            for w in config.rolling_windows:
                mean_col = f"rolling_mean_{w}"
                std_col = f"rolling_std_{w}"

                if active_group_keys:
                    grouped_target = fe_df.groupby(active_group_keys)[target_col]
                    # Shift(1) guarantees observation at index T is excluded from window at index T
                    fe_df[mean_col] = (
                        grouped_target.apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                        .reset_index(level=list(range(len(active_group_keys))), drop=True)
                    )
                    fe_df[std_col] = (
                        grouped_target.apply(lambda s: s.shift(1).rolling(window=w, min_periods=2).std())
                        .reset_index(level=list(range(len(active_group_keys))), drop=True)
                    ).fillna(0.0)
                else:
                    shifted = fe_df[target_col].shift(1)
                    fe_df[mean_col] = shifted.rolling(window=w, min_periods=1).mean()
                    fe_df[std_col] = shifted.rolling(window=w, min_periods=2).std().fillna(0.0)

                roll_cols_added.extend([mean_col, std_col])

            generated_cols.extend(roll_cols_added)
            audit.record(
                operation="rolling_feature_engineering",
                column=target_col,
                reason="Compute historical rolling moments using shift(1) to avoid lookahead contamination",
                strategy=f"shift1_rolling_windows_{config.rolling_windows}",
                rows_affected=len(fe_df),
                after_summary={"rolling_features": roll_cols_added},
            )

        # 4. Generic Business Features (Exogenous Regressors)
        if config.business_features_enabled:
            biz_cols_added: List[str] = []

            # Price Change Percentage
            price_col = mapping.price
            if price_col and price_col in fe_df.columns:
                if active_group_keys:
                    prev_price = fe_df.groupby(active_group_keys)[price_col].shift(1)
                else:
                    prev_price = fe_df[price_col].shift(1)
                fe_df["price_change_pct"] = (
                    (fe_df[price_col] - prev_price) / prev_price.replace(0, np.nan)
                ).fillna(0.0)
                biz_cols_added.append("price_change_pct")

            # Promotion Active
            promo_col = mapping.promotion
            if promo_col and promo_col in fe_df.columns:
                fe_df["promotion_active"] = fe_df[promo_col].fillna(0).astype(int)
                biz_cols_added.append("promotion_active")

            # Holiday Indicator
            hol_col = mapping.holiday
            if hol_col and hol_col in fe_df.columns:
                fe_df["holiday_indicator"] = fe_df[hol_col].fillna(0).astype(int)
                biz_cols_added.append("holiday_indicator")

            # Stockout Indicator & Inventory Pressure
            inv_col = mapping.inventory
            if inv_col and inv_col in fe_df.columns:
                fe_df["stockout_indicator"] = (fe_df[inv_col] <= 0).astype(int)
                biz_cols_added.append("stockout_indicator")

                if "rolling_mean_7" in fe_df.columns:
                    fe_df["inventory_pressure"] = (
                        fe_df[inv_col] / (fe_df["rolling_mean_7"].replace(0, np.nan) + 1.0)
                    ).fillna(1.0)
                    biz_cols_added.append("inventory_pressure")

            if biz_cols_added:
                generated_cols.extend(biz_cols_added)
                audit.record(
                    operation="business_feature_engineering",
                    reason="Derive generic business signals: price changes, stockouts, and inventory ratios",
                    strategy="historical_exogenous_transformations",
                    rows_affected=len(fe_df),
                    after_summary={"business_features": biz_cols_added},
                )

        return fe_df, generated_cols
