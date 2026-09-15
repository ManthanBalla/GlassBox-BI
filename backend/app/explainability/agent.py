"""Explainability Agent for GlassBox-BI (Phase 6).

Coordinates model interpretability across SHAP, LIME, and component decomposition:
Validates request -> Dispatches model-specific explainer -> Extracts leakage-safe
training background -> Executes local or global attribution -> Verifies model immutability ->
Assembles fidelity and audit payload -> Returns machine-readable ExplanationResult.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from backend.app.schemas.data_contract import (
    ColumnMapping,
    SYNTHETIC_RETAIL_MAPPING,
    auto_detect_column_mapping,
)
from backend.app.schemas.explainability import (
    ExplanationAuditTrail,
    ExplanationFidelity,
    ExplanationMethod,
    ExplanationResult,
    ExplanationType,
    GlobalExplanationRequest,
    LocalExplanationRequest,
)
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor
from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.explainability.base import BaseExplainer
from backend.app.explainability.shap_explainer import SHAPExplainer
from backend.app.explainability.lime_explainer import LIMEExplainer
from backend.app.explainability.prophet_explainer import ProphetComponentExplainer
from backend.app.explainability.feature_adapter import ExplainabilityFeatureAdapter
from backend.app.explainability.validation import ExplainabilityValidator


class ExplainabilityAgent:
    """Autonomous agent generating transparent local and global model explanations."""

    def __init__(
        self,
        splitter: Optional[TimeSeriesSplitter] = None,
        default_random_seed: int = 42,
    ) -> None:
        self.splitter = splitter or TimeSeriesSplitter()
        self.random_seed = default_random_seed

    def _resolve_explainer(self, method_name: str, random_seed: int) -> BaseExplainer:
        """Instantiates the appropriate explainer adapter for the resolved method."""
        m = method_name.lower().strip()
        if m == "shap":
            return SHAPExplainer(random_seed=random_seed)
        elif m == "lime":
            return LIMEExplainer(random_seed=random_seed)
        elif m == "component_based":
            return ProphetComponentExplainer(random_seed=random_seed)
        else:
            raise ValueError(f"Unsupported explanation method: '{method_name}'")

    def _load_or_resolve_dataset(
        self,
        dataset_path: Optional[str] = None,
        provided_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, ColumnMapping]:
        """Resolves processed or raw dataset source into an analysis-ready DataFrame."""
        df: Optional[pd.DataFrame] = provided_df

        if df is None:
            if dataset_path and Path(dataset_path).exists():
                df = pd.read_csv(dataset_path)
            elif Path("data/processed/synthetic/retail_processed.csv").exists():
                df = pd.read_csv("data/processed/synthetic/retail_processed.csv")
            elif Path("data/sample/retail_processed_sample.csv").exists():
                df = pd.read_csv("data/sample/retail_processed_sample.csv")
            elif Path("data/sample/retail_sample.csv").exists():
                raw_df = pd.read_csv("data/sample/retail_sample.csv")
                processor = GenericBusinessDataProcessor()
                df, _ = processor.process(raw_df, mapping=SYNTHETIC_RETAIL_MAPPING)
            else:
                raise FileNotFoundError("No valid dataset found. Please generate or process a dataset first.")

        mapping = auto_detect_column_mapping(list(df.columns))
        return df, mapping

    def _capture_model_state_snapshot(self, model: BaseForecastModel) -> Dict[str, Any]:
        """Captures a deep snapshot of model state to verify immutability."""
        snapshot: Dict[str, Any] = {
            "model_name": model.model_name,
            "training_rows": model.training_rows,
            "is_fitted": model.is_fitted,
        }

        if isinstance(model, LightGBMForecaster):
            snapshot["feature_names"] = list(model.feature_names)
            snapshot["historical_targets_len"] = len(model._historical_targets)
            if model.model is not None:
                # Number of trees
                snapshot["num_trees"] = model.model.booster_.num_trees()

        elif isinstance(model, LSTMForecaster):
            snapshot["scaler_mean"] = float(model.scaler_mean)
            snapshot["scaler_std"] = float(model.scaler_std)
            snapshot["lookback"] = model.lookback
            if model.net is not None:
                # Capture clones of parameter tensors
                snapshot["weights"] = {
                    name: param.detach().cpu().clone()
                    for name, param in model.net.named_parameters()
                }

        elif isinstance(model, ProphetForecaster):
            snapshot["training_rows"] = model.training_rows
            if model.model is not None:
                from prophet.serialize import model_to_json
                snapshot["prophet_json"] = model_to_json(model.model)

        return snapshot

    def _verify_model_immutability(self, before: Dict[str, Any], after: Dict[str, Any], model: BaseForecastModel) -> None:
        """Verifies that the model state was completely unchanged during explanation."""
        if before["model_name"] != after["model_name"]:
            raise RuntimeError("Model immutability violated: model_name changed!")

        if isinstance(model, LightGBMForecaster):
            if before["num_trees"] != after["num_trees"]:
                raise RuntimeError("Model immutability violated: LightGBM trees were modified!")
            if before["historical_targets_len"] != after["historical_targets_len"]:
                raise RuntimeError("Model immutability violated: LightGBM historical targets buffer was modified!")

        elif isinstance(model, LSTMForecaster):
            if before["scaler_mean"] != after["scaler_mean"] or before["scaler_std"] != after["scaler_std"]:
                raise RuntimeError("Model immutability violated: LSTM scaler parameters were modified!")
            for name, tensor_before in before.get("weights", {}).items():
                tensor_after = after.get("weights", {}).get(name)
                if tensor_after is None or not torch.equal(tensor_before, tensor_after):
                    raise RuntimeError(f"Model immutability violated: LSTM weight tensor '{name}' was modified!")

        elif isinstance(model, ProphetForecaster):
            if before.get("prophet_json") != after.get("prophet_json"):
                raise RuntimeError("Model immutability violated: Prophet model state changed!")

    def explain_local(
        self,
        request: LocalExplanationRequest,
        model: Optional[BaseForecastModel] = None,
        history_df: Optional[pd.DataFrame] = None,
        train_df: Optional[pd.DataFrame] = None,
    ) -> ExplanationResult:
        """Generates a structured local explanation for a specific forecast prediction."""
        start_time = time.perf_counter()
        now = datetime.now(timezone.utc)
        expl_id = f"expl_{now.strftime('%Y%m%d_%H%M%S')}"
        warnings: List[str] = []

        # 1. Resolve and Validate Explanation Method
        resolved_method = ExplainabilityValidator.validate_method_compatibility(
            model_name=request.model_name,
            method=request.method,
        )

        # 2. Resolve Data Source
        if history_df is None or train_df is None:
            full_df, mapping = self._load_or_resolve_dataset(
                dataset_path=request.dataset_path,
                provided_df=None,
            )

            # Filter by entity and product
            series_df = full_df.copy()
            if mapping.entity_id and mapping.entity_id in series_df.columns and request.entity_id:
                series_df = series_df[series_df[mapping.entity_id].astype(str) == str(request.entity_id)]
            if mapping.product_id and mapping.product_id in series_df.columns and request.product_id:
                series_df = series_df[series_df[mapping.product_id].astype(str) == str(request.product_id)]

            series_df = series_df.sort_values(by=mapping.date).reset_index(drop=True)

            # Split chronological train/val/test partitions (70% train, 15% val, 15% test)
            split_cfg = DataProcessingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
            s_train, s_val, s_test, _ = self.splitter.split(series_df, mapping=mapping, config=split_cfg)

            train_df = s_train
            history_df = pd.concat([s_train, s_val], ignore_index=True).sort_values(by=mapping.date)
            date_col = mapping.date
            target_col = mapping.target
        else:
            date_col = "date" if "date" in history_df.columns else history_df.columns[0]
            target_col = "target" if "target" in history_df.columns else history_df.columns[1]

        # 3. Resolve or Fit Model
        if model is None:
            model = self._instantiate_and_fit_model(
                model_name=request.model_name,
                history_df=history_df,
                date_col=date_col,
                target_col=target_col,
                random_seed=request.random_seed,
            )

        ExplainabilityValidator.validate_model_fitted(model)

        # 4. Capture Pre-Explanation State Snapshot for Immutability Guard
        pre_snapshot = self._capture_model_state_snapshot(model)

        # 5. Extract Prediction Feature Vector and Point Forecast
        feat_row, pred_val, res_date = ExplainabilityFeatureAdapter.extract_prediction_features(
            model=model,
            history_df=history_df,
            prediction_date=request.prediction_date,
            step_index=request.step_index,
            target_col=target_col,
            date_col=date_col,
        )

        # 6. Extract Background Reference Sample strictly from Training Data
        bg_df = ExplainabilityFeatureAdapter.extract_background_sample(
            model=model,
            train_df=train_df,
            sample_size=request.background_samples,
            random_seed=request.random_seed,
            target_col=target_col,
            date_col=date_col,
        )

        # 7. Execute Local Explainer
        explainer = self._resolve_explainer(resolved_method, random_seed=request.random_seed)
        loc_res = explainer.explain_local(
            model=model,
            feature_row=feat_row,
            prediction=pred_val,
            background_df=bg_df,
            num_lime_samples=request.num_lime_samples,
        )

        # 8. Verify Model Immutability
        post_snapshot = self._capture_model_state_snapshot(model)
        self._verify_model_immutability(pre_snapshot, post_snapshot, model)

        duration = time.perf_counter() - start_time

        # 9. Assemble Audit Trail
        top_pos = [f.feature for f in loc_res["top_positive"][:5]]
        top_neg = [f.feature for f in loc_res["top_negative"][:5]]

        audit = ExplanationAuditTrail(
            explanation_id=expl_id,
            forecast_id=None,
            model_name=model.model_name,
            model_type=model.__class__.__name__,
            entity_id=request.entity_id,
            product_id=request.product_id,
            prediction_date=res_date,
            prediction=round(pred_val, 4),
            explanation_method=resolved_method,
            explanation_type=ExplanationType.LOCAL.value,
            feature_count=len(loc_res["features"]),
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            random_seed=request.random_seed,
            sample_size=request.background_samples,
            duration_seconds=round(duration, 4),
            timestamp=now,
        )

        return ExplanationResult(
            explanation_id=expl_id,
            model_name=model.model_name,
            method=resolved_method,
            explanation_type=ExplanationType.LOCAL.value,
            prediction=round(pred_val, 4),
            prediction_date=res_date,
            base_value=loc_res.get("base_value"),
            features=loc_res["features"],
            top_positive_contributors=loc_res["top_positive"],
            top_negative_contributors=loc_res["top_negative"],
            fidelity=loc_res["fidelity"],
            audit_trail=audit,
            warnings=warnings,
            created_at=now,
        )

    def explain_global(
        self,
        request: GlobalExplanationRequest,
        model: Optional[BaseForecastModel] = None,
        history_df: Optional[pd.DataFrame] = None,
        train_df: Optional[pd.DataFrame] = None,
    ) -> ExplanationResult:
        """Computes global feature importance rankings across reference observations."""
        start_time = time.perf_counter()
        now = datetime.now(timezone.utc)
        expl_id = f"expl_global_{now.strftime('%Y%m%d_%H%M%S')}"
        warnings: List[str] = []

        # 1. Resolve and Validate Explanation Method
        resolved_method = ExplainabilityValidator.validate_method_compatibility(
            model_name=request.model_name,
            method=request.method,
        )

        # 2. Resolve Data Source
        if history_df is None or train_df is None:
            full_df, mapping = self._load_or_resolve_dataset(
                dataset_path=request.dataset_path,
                provided_df=None,
            )

            series_df = full_df.copy()
            if mapping.entity_id and mapping.entity_id in series_df.columns and request.entity_id:
                series_df = series_df[series_df[mapping.entity_id].astype(str) == str(request.entity_id)]
            if mapping.product_id and mapping.product_id in series_df.columns and request.product_id:
                series_df = series_df[series_df[mapping.product_id].astype(str) == str(request.product_id)]

            series_df = series_df.sort_values(by=mapping.date).reset_index(drop=True)

            split_cfg = DataProcessingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
            s_train, s_val, _, _ = self.splitter.split(series_df, mapping=mapping, config=split_cfg)

            train_df = s_train
            history_df = pd.concat([s_train, s_val], ignore_index=True).sort_values(by=mapping.date)
            date_col = mapping.date
            target_col = mapping.target
        else:
            date_col = "date" if "date" in history_df.columns else history_df.columns[0]
            target_col = "target" if "target" in history_df.columns else history_df.columns[1]

        # 3. Resolve or Fit Model
        if model is None:
            model = self._instantiate_and_fit_model(
                model_name=request.model_name,
                history_df=history_df,
                date_col=date_col,
                target_col=target_col,
                random_seed=request.random_seed,
            )

        ExplainabilityValidator.validate_model_fitted(model)

        # 4. Capture Pre-Explanation State Snapshot for Immutability Guard
        pre_snapshot = self._capture_model_state_snapshot(model)

        # 5. Extract Reference Data Sample (strictly from historical train partition)
        ref_df = ExplainabilityFeatureAdapter.extract_background_sample(
            model=model,
            train_df=train_df,
            sample_size=request.sample_size,
            random_seed=request.random_seed,
            target_col=target_col,
            date_col=date_col,
        )

        # 6. Execute Global Explainer
        explainer = self._resolve_explainer(resolved_method, random_seed=request.random_seed)
        glob_res = explainer.explain_global(
            model=model,
            reference_df=ref_df,
            top_k=request.top_k,
        )

        # 7. Verify Model Immutability
        post_snapshot = self._capture_model_state_snapshot(model)
        self._verify_model_immutability(pre_snapshot, post_snapshot, model)

        duration = time.perf_counter() - start_time

        top_names = [f.feature for f in glob_res["global_importance"][:5]]

        audit = ExplanationAuditTrail(
            explanation_id=expl_id,
            forecast_id=None,
            model_name=model.model_name,
            model_type=model.__class__.__name__,
            entity_id=request.entity_id,
            product_id=request.product_id,
            prediction_date=None,
            prediction=0.0,
            explanation_method=resolved_method,
            explanation_type=ExplanationType.GLOBAL.value,
            feature_count=len(glob_res["global_importance"]),
            top_positive_features=top_names,
            top_negative_features=[],
            random_seed=request.random_seed,
            sample_size=request.sample_size,
            duration_seconds=round(duration, 4),
            timestamp=now,
        )

        return ExplanationResult(
            explanation_id=expl_id,
            model_name=model.model_name,
            method=resolved_method,
            explanation_type=ExplanationType.GLOBAL.value,
            prediction=0.0,
            prediction_date=None,
            base_value=None,
            features=[],
            top_positive_contributors=[],
            top_negative_contributors=[],
            global_importance=glob_res["global_importance"],
            fidelity=glob_res["fidelity"],
            audit_trail=audit,
            warnings=warnings,
            created_at=now,
        )

    def _instantiate_and_fit_model(
        self,
        model_name: str,
        history_df: pd.DataFrame,
        date_col: str,
        target_col: str,
        random_seed: int,
    ) -> BaseForecastModel:
        """Internal helper to instantiate and fit a fresh model on history_df."""
        m_norm = model_name.lower().strip()
        if m_norm == "lightgbm":
            model = LightGBMForecaster(random_seed=random_seed)
        elif m_norm == "lstm":
            model = LSTMForecaster(random_seed=random_seed, epochs=20)
        elif m_norm == "prophet":
            model = ProphetForecaster(random_seed=random_seed)
        else:
            raise ValueError(f"Unsupported model architecture: '{model_name}'")

        model.fit(history_df, target_col=target_col, date_col=date_col)
        return model
