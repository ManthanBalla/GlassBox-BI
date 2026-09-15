"""Base Abstract Interface for All GlassBox-BI Explainers (Phase 6).

Enforces a common contract across SHAP, LIME, and component-based explainers,
guaranteeing model-agnostic local feature attribution, global feature importance,
reproducibility with deterministic seeding, and standardized attribution rankings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.app.forecasting.base import BaseForecastModel
from backend.app.schemas.explainability import (
    ExplanationFidelity,
    FeatureContribution,
    GlobalFeatureImportance,
)


class BaseExplainer(ABC):
    """Abstract contract for model interpretability and feature attribution."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    @abstractmethod
    def explain_local(
        self,
        model: BaseForecastModel,
        feature_row: pd.DataFrame,
        prediction: float,
        background_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generates local feature attributions for a single model prediction.

        Returns:
            Dict containing:
                - "base_value": float
                - "features": List[FeatureContribution]
                - "top_positive": List[FeatureContribution]
                - "top_negative": List[FeatureContribution]
                - "fidelity": ExplanationFidelity
        """
        pass

    @abstractmethod
    def explain_global(
        self,
        model: BaseForecastModel,
        reference_df: pd.DataFrame,
        top_k: Optional[int] = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculates global aggregate feature importance over a reference sample.

        Returns:
            Dict containing:
                - "global_importance": List[GlobalFeatureImportance]
                - "fidelity": ExplanationFidelity
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns structured metadata describing explainer configuration."""
        pass

    def _rank_contributions(
        self,
        contributions: Dict[str, float],
        feature_values: Optional[Dict[str, float]] = None,
    ) -> tuple[List[FeatureContribution], List[FeatureContribution], List[FeatureContribution]]:
        """Ranks local feature contributions descending by absolute magnitude.

        Returns:
            Tuple of (all_ranked, top_positive, top_negative).
        """
        sorted_feats = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        all_contributions: List[FeatureContribution] = []
        positive_contributors: List[FeatureContribution] = []
        negative_contributors: List[FeatureContribution] = []

        for rank_idx, (fname, contrib) in enumerate(sorted_feats, start=1):
            val = feature_values.get(fname) if feature_values else None
            abs_c = round(abs(contrib), 4)
            c_round = round(contrib, 4)

            fc = FeatureContribution(
                feature=fname,
                value=round(val, 4) if val is not None else None,
                contribution=c_round,
                shap_value=c_round,
                absolute_contribution=abs_c,
                direction="positive" if c_round > 1e-5 else ("negative" if c_round < -1e-5 else "neutral"),
                rank=rank_idx,
            )
            all_contributions.append(fc)

            if c_round > 1e-5:
                positive_contributors.append(fc)
            elif c_round < -1e-5:
                negative_contributors.append(fc)

        # Sort positive descending by positive impact
        pos_sorted = sorted(positive_contributors, key=lambda x: x.contribution, reverse=True)
        # Sort negative descending by magnitude of negative impact
        neg_sorted = sorted(negative_contributors, key=lambda x: x.contribution, reverse=False)

        return all_contributions, pos_sorted, neg_sorted

    def _rank_global_importance(
        self,
        importance_scores: Dict[str, float],
        top_k: Optional[int] = 20,
    ) -> List[GlobalFeatureImportance]:
        """Ranks global feature importance descending by aggregate score."""
        sorted_items = sorted(
            importance_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        total_importance = sum(importance_scores.values()) if importance_scores else 0.0

        ranked_list: List[GlobalFeatureImportance] = []
        for rank_idx, (fname, score) in enumerate(sorted_items, start=1):
            norm_score = (score / total_importance) if total_importance > 1e-9 else 0.0
            ranked_list.append(
                GlobalFeatureImportance(
                    feature=fname,
                    importance_score=round(score, 4),
                    rank=rank_idx,
                    normalized_importance=round(norm_score, 4),
                )
            )

        if top_k is not None and top_k > 0:
            return ranked_list[:top_k]
        return ranked_list
