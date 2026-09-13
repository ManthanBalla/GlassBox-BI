"""Data Processing & Ingestion Module for GlassBox-BI.

Provides source-agnostic dataset ingestion, multi-dimensional validation,
temporal leakage detection, data profiling, and explainable quality scoring.
"""

from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector
from backend.app.data_processing.quality import DataQualityScorer
from backend.app.data_processing.profiling import DataProfiler
from backend.app.data_processing.ingestion import DatasetIngestionService
from backend.app.data_processing.processing import (
    AuditTrailTracker,
    DataCleaner,
    DuplicateHandler,
    InvalidValueHandler,
    MissingValueHandler,
    OutlierDetector,
    TimeSeriesAnalyzer,
    FeatureEngineer,
    TimeSeriesSplitter,
    GenericBusinessDataProcessor,
)

__all__ = [
    "DataValidator",
    "TemporalLeakageDetector",
    "DataQualityScorer",
    "DataProfiler",
    "DatasetIngestionService",
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

