"""High-Performance Statistical Data Profiler for GlassBox-BI.

Generates structured profiles, numeric percentiles, outlier metrics, and time-series
frequencies from tabular business datasets.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import DatasetProfile, NumericSummary


class DataProfiler:
    """Profiles tabular datasets to extract dimensional metrics, cardinalities, and statistical bounds."""

    def profile(self, df: pd.DataFrame, mapping: ColumnMapping) -> DatasetProfile:
        """Executes statistical profiling across the dataframe."""
        row_count = len(df)
        col_count = len(df.columns)
        columns = list(df.columns)
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Missing percentages
        missing_percentages = {}
        for col in columns:
            null_count = int(df[col].isna().sum())
            missing_percentages[col] = round((null_count / max(1, row_count)) * 100.0, 2)

        # Date diagnostics & frequency inference
        date_min: Optional[str] = None
        date_max: Optional[str] = None
        inferred_freq: Optional[str] = None

        if mapping.date in df.columns:
            date_series = pd.to_datetime(df[mapping.date], errors="coerce").dropna().sort_values()
            if len(date_series) > 0:
                date_min = date_series.min().strftime("%Y-%m-%d")
                date_max = date_series.max().strftime("%Y-%m-%d")

                # Incur frequency on unique timestamps
                unique_dates = pd.Series(date_series.unique()).sort_values()
                if len(unique_dates) > 2:
                    diffs = unique_dates.diff().dropna()
                    median_days = diffs.dt.total_seconds().median() / 86400.0
                    if 0.8 <= median_days <= 1.2:
                        inferred_freq = "Daily (D)"
                    elif 6.0 <= median_days <= 8.0:
                        inferred_freq = "Weekly (W)"
                    elif 27.0 <= median_days <= 32.0:
                        inferred_freq = "Monthly (M)"
                    elif 85.0 <= median_days <= 95.0:
                        inferred_freq = "Quarterly (Q)"
                    elif 360.0 <= median_days <= 370.0:
                        inferred_freq = "Yearly (Y)"
                    else:
                        inferred_freq = f"Irregular (~{median_days:.1f} days)"

        # Cardinalities
        unique_entities = int(df[mapping.entity_id].nunique()) if mapping.entity_id in df.columns else 0
        unique_products = int(df[mapping.product_id].nunique()) if mapping.product_id and mapping.product_id in df.columns else 0
        unique_categories = int(df[mapping.category].nunique()) if mapping.category and mapping.category in df.columns else 0
        unique_regions = int(df[mapping.region].nunique()) if mapping.region and mapping.region in df.columns else 0

        # Numeric Summaries
        numeric_summaries: Dict[str, NumericSummary] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

            numeric_summaries[col] = NumericSummary(
                count=int(len(series)),
                min=round(float(series.min()), 2),
                max=round(float(series.max()), 2),
                mean=round(float(series.mean()), 2),
                median=round(float(series.median()), 2),
                std=round(float(series.std() if len(series) > 1 else 0.0), 2),
                q25=round(q25, 2),
                q75=round(q75, 2),
                outliers_iqr=outlier_count,
            )

        target_summary = numeric_summaries.get(mapping.target)

        return DatasetProfile(
            row_count=row_count,
            column_count=col_count,
            columns=columns,
            data_types=data_types,
            date_min=date_min,
            date_max=date_max,
            date_frequency=inferred_freq,
            unique_entities=unique_entities,
            unique_products=unique_products,
            unique_categories=unique_categories,
            unique_regions=unique_regions,
            missing_percentages=missing_percentages,
            numeric_summary=numeric_summaries,
            target_summary=target_summary,
        )
