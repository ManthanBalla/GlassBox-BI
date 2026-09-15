"""Explainability and Feature Attribution Engine for GlassBox-BI (Phase 6).

Provides model-agnostic and model-specific local and global explanations:
- SHAP TreeExplainer (LightGBM) and sequence attribution (LSTM)
- LIME local linear surrogates with reproducible random seeds
- Component-based additive decomposition (Prophet)
- Zero-leakage background distributions sampled strictly from training data
- Pre/post explanation model immutability verification
"""

from backend.app.explainability.base import BaseExplainer
from backend.app.explainability.shap_explainer import SHAPExplainer
from backend.app.explainability.lime_explainer import LIMEExplainer
from backend.app.explainability.prophet_explainer import ProphetComponentExplainer
from backend.app.explainability.feature_adapter import ExplainabilityFeatureAdapter
from backend.app.explainability.validation import (
    ExplainabilityValidator,
    MODEL_COMPATIBILITY,
)
from backend.app.explainability.agent import ExplainabilityAgent

__all__ = [
    "BaseExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "ProphetComponentExplainer",
    "ExplainabilityFeatureAdapter",
    "ExplainabilityValidator",
    "MODEL_COMPATIBILITY",
    "ExplainabilityAgent",
]
