"""Time-Series Ordering, Frequency Inspection, and Gap Analysis for GlassBox-BI.

Ensures strict chronological sorting, infers observation cadences, and computes
detailed continuity diagnostics per series group without synthetic forward interpolation.
"""

from typing import List, Tuple
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig, TimeSeriesIntegrityReport
from backend.app.data_processing.processing.audit import AuditTrailTracker


class TimeSeriesAnalyzer:
    """Analyzes and enforces temporal integrity across business time-series."""

    def analyze_and_sort(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> Tuple[pd.DataFrame, TimeSeriesIntegrityReport]:
        """Sorts data chronologically and produces a TimeSeriesIntegrityReport.

        Args:
            df: Dataframe after missing value handling.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Tuple of (sorted_df, TimeSeriesIntegrityReport).
        """
        # 1. Determine Hierarchy Columns for Sorting
        sort_cols = [mapping.date, mapping.entity_id]
        if mapping.product_id and mapping.product_id in df.columns:
            sort_cols.append(mapping.product_id)

        active_sort_cols = [c for c in sort_cols if c in df.columns]

        # 2. Enforce Chronological Sorting
        sorted_df = df.sort_values(by=active_sort_cols, ascending=True).reset_index(drop=True)

        audit.record(
            operation="chronological_sorting",
            reason="Preserve strict temporal order required for time-series modeling",
            strategy="sort_by_" + "_".join(active_sort_cols),
            rows_affected=len(sorted_df),
            after_summary={"sorted_columns": active_sort_cols},
        )

        # 3. Analyze Series Group Continuity & Gaps
        date_col = mapping.date
        entity_col = mapping.entity_id

        # Grouping keys
        group_keys = [entity_col]
        if mapping.product_id and mapping.product_id in sorted_df.columns:
            group_keys.append(mapping.product_id)
        active_group_keys = [k for k in group_keys if k in sorted_df.columns]

        dates = pd.to_datetime(sorted_df[date_col]).dropna()
        min_date_str = str(dates.min().date()) if len(dates) > 0 else None
        max_date_str = str(dates.max().date()) if len(dates) > 0 else None

        # Infer Global Frequency
        inferred_freq = "D"
        if len(dates) > 1:
            try:
                unique_dates = pd.Series(dates.unique()).sort_values()
                inferred = pd.infer_freq(unique_dates)
                if inferred:
                    inferred_freq = inferred
                else:
                    median_diff_days = (unique_dates.diff().dt.total_seconds() / 86400).median()
                    if median_diff_days == 7:
                        inferred_freq = "W"
                    elif median_diff_days == 1:
                        inferred_freq = "D"
            except Exception:
                inferred_freq = "D"

        # Per-series gap analysis
        series_with_gaps = 0
        total_missing_periods = 0
        largest_gap = 0
        series_lengths: List[int] = []

        if active_group_keys:
            grouped = sorted_df.groupby(active_group_keys)
            total_series_count = len(grouped)

            for _, group in grouped:
                s_dates = pd.to_datetime(group[date_col]).sort_values().drop_duplicates()
                series_len = len(s_dates)
                series_lengths.append(series_len)

                if series_len > 1:
                    day_diffs = (s_dates.diff().dt.total_seconds() / 86400).dropna()
                    # Expected cadence interval in days
                    expected_cadence = 7.0 if inferred_freq.startswith("W") else 1.0
                    gaps = day_diffs[day_diffs > (expected_cadence * 1.5)]
                    
                    if len(gaps) > 0:
                        series_with_gaps += 1
                        missing_count = int(sum(round((g - expected_cadence) / expected_cadence) for g in gaps))
                        total_missing_periods += missing_count
                        curr_max_gap = int(max(round((g - expected_cadence) / expected_cadence) for g in gaps))
                        if curr_max_gap > largest_gap:
                            largest_gap = curr_max_gap
        else:
            total_series_count = 1
            series_lengths = [len(sorted_df)]

        min_len = min(series_lengths) if series_lengths else 0
        max_len = max(series_lengths) if series_lengths else 0

        # Total expected observations across all series
        expected_total = sum(series_lengths) + total_missing_periods
        continuity_pct = round((sum(series_lengths) / max(1, expected_total)) * 100.0, 2)

        report = TimeSeriesIntegrityReport(
            expected_frequency=inferred_freq,
            total_series_count=total_series_count,
            series_with_gaps=series_with_gaps,
            total_missing_periods=total_missing_periods,
            largest_gap_periods=largest_gap,
            continuity_percentage=continuity_pct,
            min_date=min_date_str,
            max_date=max_date_str,
            min_series_length=min_len,
            max_series_length=max_len,
        )

        audit.record(
            operation="time_series_gap_analysis",
            reason="Evaluate temporal regularity and observation cadence per series",
            strategy=f"inferred_frequency_{inferred_freq}",
            rows_affected=len(sorted_df),
            after_summary={
                "inferred_frequency": inferred_freq,
                "total_series": total_series_count,
                "series_with_gaps": series_with_gaps,
                "continuity_percentage": continuity_pct,
            },
        )

        return sorted_df, report
