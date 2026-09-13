"""Data Processing Agent Sub-Package for GlassBox-BI (Phase 3).

Exposes the GenericBusinessDataProcessor and specialized preprocessing modules:
cleaning, missing value imputation, duplicate resolution, outlier profiling,
time-series gap analysis, feature engineering, and walk-forward splitting.
"""

from backend.app.data_processing.processing.audit import AuditTrailTracker
from backend.app.data_processing.processing.cleaning import DataCleaner
from backend.app.data_processing.processing.duplicates import DuplicateHandler
from backend.app.data_processing.processing.invalid_values import InvalidValueHandler
from backend.app.data_processing.processing.missing_values import MissingValueHandler
from backend.app.data_processing.processing.outliers import OutlierDetector
from backend.app.data_processing.processing.time_series import TimeSeriesAnalyzer
from backend.app.data_processing.processing.feature_engineering import FeatureEngineer
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor

__all__ = [
    "AuditTrailTracker",
    "DataCleaner",
    "DuplicateHandler",
    "InvalidValueHandler",
    "MissingValueHandler",
    "OutlierDetector",
    "TimeSeriesAnalyzer",
    "FeatureEngineer",
    "TimeSeriesSplitter",
    "GenericBusinessDataProcessor",
]
