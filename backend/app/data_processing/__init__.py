"""Data Processing & Ingestion Module for GlassBox-BI.

Provides source-agnostic dataset ingestion, multi-dimensional validation,
temporal leakage detection, data profiling, and explainable quality scoring.
"""

from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector
from backend.app.data_processing.quality import DataQualityScorer
from backend.app.data_processing.profiling import DataProfiler
from backend.app.data_processing.ingestion import DatasetIngestionService

__all__ = [
    "DataValidator",
    "TemporalLeakageDetector",
    "DataQualityScorer",
    "DataProfiler",
    "DatasetIngestionService",
]
