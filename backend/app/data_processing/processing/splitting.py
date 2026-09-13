"""Temporal Walk-Forward Splitting Utility for GlassBox-BI.

Prepares chronological train, validation, and test partitions for future forecasting.
Strict Rule: NO models are trained in Phase 3. This is a pure data preparation utility.
NEVER shuffles time-series observations.
"""

from typing import Optional, Tuple
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig, TemporalSplitMetadata
from backend.app.data_processing.processing.audit import AuditTrailTracker


class TimeSeriesSplitter:
    """Partitions time-series datasets chronologically without lookahead leakage."""

    def split(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: Optional[AuditTrailTracker] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TemporalSplitMetadata]:
        """Performs a strictly chronological walk-forward split.

        Args:
            df: Chronologically sorted and feature-engineered dataframe.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Optional audit trail tracker.

        Returns:
            Tuple of (train_df, val_df, test_df, TemporalSplitMetadata).
        """
        date_col = mapping.date
        if date_col not in df.columns or len(df) == 0:
            empty_meta = TemporalSplitMetadata(
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                train_rows=0,
                val_rows=0,
                test_rows=0,
                temporal_order_verified=True,
            )
            return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy(), empty_meta

        # Extract sorted unique observation dates
        unique_dates = pd.Series(pd.to_datetime(df[date_col]).unique()).sort_values().reset_index(drop=True)
        n_dates = len(unique_dates)

        if n_dates < 3:
            # Not enough distinct dates for 3-way split; fallback to row-based contiguous slice
            n_rows = len(df)
            train_cut = int(n_rows * config.train_ratio)
            val_cut = int(n_rows * (config.train_ratio + config.val_ratio))

            train_df = df.iloc[:train_cut].copy()
            val_df = df.iloc[train_cut:val_cut].copy()
            test_df = df.iloc[val_cut:].copy()

            meta = TemporalSplitMetadata(
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                train_rows=len(train_df),
                val_rows=len(val_df),
                test_rows=len(test_df),
                temporal_order_verified=True,
            )
            return train_df, val_df, test_df, meta

        # Calculate contiguous chronological cutoff indices by date
        train_date_count = max(1, int(round(n_dates * config.train_ratio)))
        val_date_count = max(1, int(round(n_dates * config.val_ratio)))
        if train_date_count + val_date_count >= n_dates:
            # Adjust so test has at least 1 date
            train_date_count = max(1, n_dates - 2)
            val_date_count = 1

        train_cutoff_date = unique_dates.iloc[train_date_count - 1]
        val_start_date = unique_dates.iloc[train_date_count]
        val_cutoff_date = unique_dates.iloc[train_date_count + val_date_count - 1]
        test_start_date = unique_dates.iloc[train_date_count + val_date_count]
        test_cutoff_date = unique_dates.iloc[-1]

        # Partition based on date boundaries
        dates_series = pd.to_datetime(df[date_col])
        train_mask = dates_series <= train_cutoff_date
        val_mask = (dates_series >= val_start_date) & (dates_series <= val_cutoff_date)
        test_mask = dates_series >= test_start_date

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        temporal_verified = train_cutoff_date <= val_start_date <= val_cutoff_date <= test_start_date

        metadata = TemporalSplitMetadata(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_start_date=str(unique_dates.iloc[0].date()),
            train_end_date=str(train_cutoff_date.date()),
            val_start_date=str(val_start_date.date()),
            val_end_date=str(val_cutoff_date.date()),
            test_start_date=str(test_start_date.date()),
            test_end_date=str(test_cutoff_date.date()),
            temporal_order_verified=bool(temporal_verified),
        )

        if audit:
            audit.record(
                operation="temporal_walk_forward_split",
                reason="Generate chronological boundaries without temporal leakage or random shuffling",
                strategy=f"ratio_{config.train_ratio}_{config.val_ratio}_{config.test_ratio}",
                rows_affected=len(df),
                after_summary={
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "train_end": metadata.train_end_date,
                    "val_start": metadata.val_start_date,
                    "test_start": metadata.test_start_date,
                },
            )

        return train_df, val_df, test_df, metadata
