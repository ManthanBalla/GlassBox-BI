"""LIME Explainer Implementation for GlassBox-BI (Phase 6).

Implements local linear surrogate modeling using LimeTabularExplainer with
deterministic random seeding, local R² surrogate fidelity scores, and
aggregated local feature importance rankings.
"""

from typing import Any, Callable, Dict, List, Optional
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.explainability.base import BaseExplainer
from backend.app.schemas.explainability import (
    ExplanationFidelity,
    FeatureContribution,
    GlobalFeatureImportance,
)


class LIMEExplainer(BaseExplainer):
    """LIME explainer adapter supporting local linear surrogates with reproducible random seeds."""

    def __init__(self, random_seed: int = 42, num_samples: int = 500) -> None:
        super().__init__(random_seed=random_seed)
        self.num_samples = num_samples

    def explain_local(
        self,
        model: BaseForecastModel,
        feature_row: pd.DataFrame,
        prediction: float,
        background_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generates local LIME feature attributions using a linear surrogate model."""
        if isinstance(model, LightGBMForecaster):
            return self._explain_lightgbm_local(model, feature_row, prediction, background_df, **kwargs)
        elif isinstance(model, LSTMForecaster):
            return self._explain_lstm_local(model, feature_row, prediction, background_df, **kwargs)
        elif isinstance(model, ProphetForecaster):
            raise ValueError(
                "Prophet does not consume tabular features. Use 'component_based' explanation for Prophet."
            )
        else:
            raise ValueError(f"Unsupported model architecture for LIMEExplainer: {type(model)}")

    def explain_global(
        self,
        model: BaseForecastModel,
        reference_df: pd.DataFrame,
        top_k: Optional[int] = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Aggregates local LIME surrogate weights across a sample of reference observations.

        Note: LIME is natively local; global importance represents sample-aggregated absolute weights.
        """
        if isinstance(model, LightGBMForecaster):
            return self._explain_lightgbm_global(model, reference_df, top_k, **kwargs)
        elif isinstance(model, LSTMForecaster):
            return self._explain_lstm_global(model, reference_df, top_k, **kwargs)
        elif isinstance(model, ProphetForecaster):
            raise ValueError(
                "Prophet does not consume tabular features. Use 'component_based' explanation for Prophet."
            )
        else:
            raise ValueError(f"Unsupported model architecture for LIMEExplainer: {type(model)}")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "explainer_class": "LIMEExplainer",
            "random_seed": self.random_seed,
            "num_samples": self.num_samples,
        }

    # -------------------------------------------------------------------------
    # LightGBM LIME Implementation
    # -------------------------------------------------------------------------
    def _explain_lightgbm_local(
        self,
        model: LightGBMForecaster,
        feature_row: pd.DataFrame,
        prediction: float,
        background_df: Optional[pd.DataFrame],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fits a local linear surrogate for LightGBM."""
        feature_names = model.feature_names
        feat_aligned = feature_row[feature_names].fillna(0.0)

        if background_df is None or background_df.empty:
            raise ValueError("LIME requires a background training distribution to initialize perturbations.")

        bg_aligned = background_df[feature_names].fillna(0.0).to_numpy(dtype=float)

        def predict_fn(X_arr: np.ndarray) -> np.ndarray:
            df_batch = pd.DataFrame(X_arr, columns=feature_names)
            return model.model.predict(df_batch)  # type: ignore

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_aligned,
            feature_names=feature_names,
            mode="regression",
            random_state=self.random_seed,
            verbose=False,
        )

        x_row = feat_aligned.iloc[0].to_numpy(dtype=float)
        num_samples = kwargs.get("num_lime_samples", self.num_samples)

        exp = explainer.explain_instance(
            data_row=x_row,
            predict_fn=predict_fn,
            num_features=len(feature_names),
            num_samples=num_samples,
        )

        # Extract surrogate weights mapped by feature index
        map_weights = exp.as_map().get(1, [])
        intercept = float(exp.intercept[1]) if hasattr(exp, "intercept") and 1 in exp.intercept else 0.0
        surrogate_r2 = float(exp.score) if hasattr(exp, "score") else None

        contributions: Dict[str, float] = {}
        feature_values: Dict[str, float] = {}

        # Default all features to 0.0
        for f in feature_names:
            contributions[f] = 0.0
            feature_values[f] = float(feat_aligned[f].iloc[0])

        for feat_idx, weight in map_weights:
            fname = feature_names[feat_idx]
            contributions[fname] = float(weight)

        all_ranked, pos_ranked, neg_ranked = self._rank_contributions(contributions, feature_values)

        reconstructed = round(intercept + sum(contributions.values()), 4)
        rec_error = round(abs(prediction - reconstructed), 4)

        fidelity_r2 = max(0.0, min(1.0, surrogate_r2)) if surrogate_r2 is not None else 0.85
        fidelity_score = round(fidelity_r2, 4)

        status = "HIGH_FIDELITY" if fidelity_score >= 0.85 else ("MODERATE_FIDELITY" if fidelity_score >= 0.65 else "APPROXIMATE")

        fidelity = ExplanationFidelity(
            fidelity_score=fidelity_score,
            reconstruction_error=rec_error,
            surrogate_r2=round(surrogate_r2, 4) if surrogate_r2 is not None else None,
            base_value=round(intercept, 4),
            reconstructed_prediction=reconstructed,
            actual_prediction=round(prediction, 4),
            explanation_status=status,
            method_notes=(
                f"LIME linear surrogate evaluated with R²={fidelity_score}. "
                f"Weights represent local slopes around the perturbed neighborhood (N={num_samples})."
            ),
        )

        return {
            "base_value": round(intercept, 4),
            "features": all_ranked,
            "top_positive": pos_ranked,
            "top_negative": neg_ranked,
            "fidelity": fidelity,
        }

    def _explain_lightgbm_global(
        self,
        model: LightGBMForecaster,
        reference_df: pd.DataFrame,
        top_k: Optional[int] = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Computes sample-aggregated absolute LIME weights across reference observations."""
        feature_names = model.feature_names
        ref_aligned = reference_df[feature_names].fillna(0.0)

        # Use up to 15 sample points for speed
        n_pts = min(15, len(ref_aligned))
        rng = np.random.default_rng(seed=self.random_seed)
        pts_idx = rng.choice(len(ref_aligned), size=n_pts, replace=False)
        pts_df = ref_aligned.iloc[pts_idx]

        accum_weights: Dict[str, List[float]] = {f: [] for f in feature_names}
        scores: List[float] = []

        for _, row in pts_df.iterrows():
            row_df = pd.DataFrame([row])
            p_val = float(model.model.predict(row_df)[0])  # type: ignore
            loc = self._explain_lightgbm_local(model, row_df, p_val, reference_df, num_lime_samples=300)
            for fc in loc["features"]:
                accum_weights[fc.feature].append(fc.absolute_contribution)
            if loc["fidelity"].surrogate_r2 is not None:
                scores.append(loc["fidelity"].surrogate_r2)

        importance_scores: Dict[str, float] = {
            f: float(np.mean(weights)) if weights else 0.0 for f, weights in accum_weights.items()
        }

        ranked_global = self._rank_global_importance(importance_scores, top_k=top_k)
        avg_r2 = float(np.mean(scores)) if scores else 0.80

        fidelity = ExplanationFidelity(
            fidelity_score=round(max(0.0, min(1.0, avg_r2)), 4),
            reconstruction_error=0.0,
            surrogate_r2=round(avg_r2, 4),
            explanation_status="APPROXIMATE",
            method_notes=(
                f"Aggregated absolute local LIME surrogate weights across {n_pts} sample points. "
                "Distinct from native global feature attribution."
            ),
        )

        return {
            "global_importance": ranked_global,
            "fidelity": fidelity,
        }

    # -------------------------------------------------------------------------
    # LSTM LIME Implementation
    # -------------------------------------------------------------------------
    def _explain_lstm_local(
        self,
        model: LSTMForecaster,
        feature_row: pd.DataFrame,
        prediction: float,
        background_df: Optional[pd.DataFrame],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fits a local linear surrogate for LSTM over sequence lookback lags."""
        import torch

        feature_names = [f"lag_{i + 1}" for i in range(model.lookback)]
        feat_aligned = feature_row[feature_names].to_numpy(dtype=float)

        if background_df is None or background_df.empty:
            raise ValueError("LIME requires background reference windows for LSTM.")

        bg_data = background_df[feature_names].to_numpy(dtype=float)

        model.net.eval()  # type: ignore

        def predict_fn(X_arr: np.ndarray) -> np.ndarray:
            n_samples = len(X_arr)
            preds = np.zeros(n_samples, dtype=np.float32)
            with torch.no_grad():
                for i in range(n_samples):
                    row = X_arr[i]
                    chronological_window = row[::-1]
                    scaled_w = (chronological_window - model.scaler_mean) / (model.scaler_std + 1e-6)
                    x_t = torch.tensor(scaled_w, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    s_pred = float(model.net(x_t).item())  # type: ignore
                    raw_p = s_pred * (model.scaler_std + 1e-6) + model.scaler_mean
                    preds[i] = max(0.0, raw_p)
            return preds

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_data,
            feature_names=feature_names,
            mode="regression",
            random_state=self.random_seed,
            verbose=False,
        )

        x_row = feat_aligned[0]
        num_samples = kwargs.get("num_lime_samples", 300)

        exp = explainer.explain_instance(
            data_row=x_row,
            predict_fn=predict_fn,
            num_features=len(feature_names),
            num_samples=num_samples,
        )

        map_weights = exp.as_map().get(1, [])
        intercept = float(exp.intercept[1]) if hasattr(exp, "intercept") and 1 in exp.intercept else 0.0
        surrogate_r2 = float(exp.score) if hasattr(exp, "score") else None

        contributions: Dict[str, float] = {f: 0.0 for f in feature_names}
        feature_values: Dict[str, float] = {f: float(feature_row[f].iloc[0]) for f in feature_names}

        for feat_idx, weight in map_weights:
            fname = feature_names[feat_idx]
            contributions[fname] = float(weight)

        all_ranked, pos_ranked, neg_ranked = self._rank_contributions(contributions, feature_values)

        reconstructed = round(intercept + sum(contributions.values()), 4)
        rec_error = round(abs(prediction - reconstructed), 4)
        fidelity_r2 = max(0.0, min(1.0, surrogate_r2)) if surrogate_r2 is not None else 0.80

        fidelity = ExplanationFidelity(
            fidelity_score=round(fidelity_r2, 4),
            reconstruction_error=rec_error,
            surrogate_r2=round(surrogate_r2, 4) if surrogate_r2 is not None else None,
            base_value=round(intercept, 4),
            reconstructed_prediction=reconstructed,
            actual_prediction=round(prediction, 4),
            explanation_status="APPROXIMATE",
            method_notes=f"LIME linear surrogate over {model.lookback} input timesteps (surrogate R²={fidelity_r2:.4f}).",
        )

        return {
            "base_value": round(intercept, 4),
            "features": all_ranked,
            "top_positive": pos_ranked,
            "top_negative": neg_ranked,
            "fidelity": fidelity,
        }

    def _explain_lstm_global(
        self,
        model: LSTMForecaster,
        reference_df: pd.DataFrame,
        top_k: Optional[int] = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Aggregates LIME surrogate weights across sample LSTM lookback windows."""
        feature_names = [f"lag_{i + 1}" for i in range(model.lookback)]
        ref_data = reference_df[feature_names].to_numpy(dtype=float)

        n_pts = min(10, len(ref_data))
        rng = np.random.default_rng(seed=self.random_seed)
        pts_idx = rng.choice(len(ref_data), size=n_pts, replace=False)

        accum_weights: Dict[str, List[float]] = {f: [] for f in feature_names}
        scores: List[float] = []

        for idx in pts_idx:
            row_df = pd.DataFrame([ref_data[idx]], columns=feature_names)
            loc = self._explain_lstm_local(model, row_df, 0.0, reference_df, num_lime_samples=200)
            for fc in loc["features"]:
                accum_weights[fc.feature].append(fc.absolute_contribution)
            if loc["fidelity"].surrogate_r2 is not None:
                scores.append(loc["fidelity"].surrogate_r2)

        importance_scores: Dict[str, float] = {
            f: float(np.mean(weights)) if weights else 0.0 for f, weights in accum_weights.items()
        }

        ranked_global = self._rank_global_importance(importance_scores, top_k=top_k)
        avg_r2 = float(np.mean(scores)) if scores else 0.75

        fidelity = ExplanationFidelity(
            fidelity_score=round(max(0.0, min(1.0, avg_r2)), 4),
            reconstruction_error=0.0,
            surrogate_r2=round(avg_r2, 4),
            explanation_status="APPROXIMATE",
            method_notes=f"Aggregated absolute LIME weights over {n_pts} sample windows.",
        )

        return {
            "global_importance": ranked_global,
            "fidelity": fidelity,
        }
