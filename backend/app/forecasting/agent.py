"""Generic Forecasting Agent for GlassBox-BI (Phase 4).

Orchestrates multi-model forecasting competition across Prophet, LightGBM, and LSTM:
Validates series history -> Splits chronological train/val partitions ->
Fits candidates -> Evaluates validation metrics -> Ranks and selects best model ->
Refits winner on (train + val) -> Generates future forecast with prediction intervals ->
Persists model artifact -> Returns structured, machine-readable ForecastResult.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import (
    ColumnMapping,
    SYNTHETIC_RETAIL_MAPPING,
    auto_detect_column_mapping,
)
from backend.app.schemas.forecasting import (
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
    ForecastStatus,
    ForecastingConfig,
    ModelEvaluationScore,
    ModelMetadata,
    ModelSelectionResult,
    SelectionMetric,
)
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor
from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.persistence import save_forecaster


class GenericForecastingAgent:
    """Universal, organization-agnostic agent coordinating time-series forecasting models."""

    def __init__(
        self,
        default_config: Optional[ForecastingConfig] = None,
        splitter: Optional[TimeSeriesSplitter] = None,
        persistence_dir: Union[str, Path] = "models/saved",
    ) -> None:
        self.config = default_config or ForecastingConfig()
        self.splitter = splitter or TimeSeriesSplitter()
        self.persistence_dir = Path(persistence_dir)

    def _instantiate_model(self, model_name: str, config: ForecastingConfig) -> BaseForecastModel:
        """Instantiates a fresh model adapter for the specified candidate architecture."""
        m_name = model_name.lower().strip()
        if m_name == "prophet":
            return ProphetForecaster(
                model_name="prophet",
                random_seed=config.random_seed,
                yearly_seasonality=config.prophet_yearly_seasonality,
                weekly_seasonality=config.prophet_weekly_seasonality,
                daily_seasonality=config.prophet_daily_seasonality,
                interval_width=config.confidence_level,
            )
        elif m_name == "lightgbm":
            return LightGBMForecaster(
                model_name="lightgbm",
                random_seed=config.random_seed,
                n_estimators=config.lgb_n_estimators,
                learning_rate=config.lgb_learning_rate,
                max_depth=config.lgb_max_depth,
                num_leaves=config.lgb_num_leaves,
            )
        elif m_name == "lstm":
            return LSTMForecaster(
                model_name="lstm",
                random_seed=config.random_seed,
                lookback=config.lstm_lookback,
                hidden_dim=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
                learning_rate=config.lstm_learning_rate,
                epochs=config.lstm_epochs,
                batch_size=config.lstm_batch_size,
                patience=config.lstm_patience,
            )
        else:
            raise ValueError(f"Unsupported candidate forecasting model: {model_name}")

    def _load_or_resolve_dataset(
        self,
        request: ForecastRequest,
        provided_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, ColumnMapping]:
        """Resolves processed or raw dataset source into an analysis-ready DataFrame."""
        df: Optional[pd.DataFrame] = provided_df

        if df is None:
            # Check provided custom path
            if request.dataset_path and Path(request.dataset_path).exists():
                df = pd.read_csv(request.dataset_path)
            # Default to processed synthetic retail dataset if available
            elif Path("data/processed/synthetic/retail_processed.csv").exists():
                df = pd.read_csv("data/processed/synthetic/retail_processed.csv")
            # Fallback to sample processed dataset
            elif Path("data/sample/retail_processed_sample.csv").exists():
                df = pd.read_csv("data/sample/retail_processed_sample.csv")
            # Fallback to raw sample (execute Phase 3 processor)
            elif Path("data/sample/retail_sample.csv").exists():
                raw_df = pd.read_csv("data/sample/retail_sample.csv")
                processor = GenericBusinessDataProcessor()
                df, _ = processor.process(raw_df, mapping=SYNTHETIC_RETAIL_MAPPING)
            else:
                raise FileNotFoundError(
                    "No valid dataset found. Please generate or process a dataset first."
                )

        mapping = auto_detect_column_mapping(list(df.columns))
        return df, mapping

    def run_forecast(
        self,
        request: ForecastRequest,
        df: Optional[pd.DataFrame] = None,
        custom_config: Optional[ForecastingConfig] = None,
    ) -> ForecastResult:
        """Executes candidate model competition, selection, and future forecast generation.

        Args:
            request: ForecastRequest specifying series, horizon, and candidate architectures.
            df: Optional in-memory DataFrame (loaded from storage if omitted).
            custom_config: Optional runtime configuration override.

        Returns:
            Structured ForecastResult machine-readable payload.
        """
        now = datetime.now(timezone.utc)
        forecast_id = f"fc_{now.strftime('%Y%m%d_%H%M%S')}"
        effective_config = custom_config or self.config
        warnings: List[str] = []

        # 1. Resolve Dataset & Column Mapping
        loaded_df, mapping = self._load_or_resolve_dataset(request, provided_df=df)
        date_col = request.date_column if request.date_column in loaded_df.columns else mapping.date
        target_col = request.target_column if request.target_column in loaded_df.columns else mapping.target
        entity_col = mapping.entity_id
        product_col = mapping.product_id

        # 2. Filter Series by Entity and Product
        series_df = loaded_df.copy()
        resolved_entity = request.entity_id
        resolved_product = request.product_id

        if entity_col and entity_col in series_df.columns:
            if not resolved_entity:
                resolved_entity = str(series_df[entity_col].iloc[0])
                warnings.append(f"entity_id was omitted; defaulting to first series entity '{resolved_entity}'.")
            series_df = series_df[series_df[entity_col].astype(str) == str(resolved_entity)]

        if product_col and product_col in series_df.columns:
            if not resolved_product:
                resolved_product = str(series_df[product_col].iloc[0])
                warnings.append(f"product_id was omitted; defaulting to first product '{resolved_product}'.")
            series_df = series_df[series_df[product_col].astype(str) == str(resolved_product)]

        series_df = series_df.sort_values(by=date_col).reset_index(drop=True)
        total_series_rows = len(series_df)

        if total_series_rows < 15:
            # Insufficient series observations to split and train
            return ForecastResult(
                forecast_id=forecast_id,
                dataset_id=request.dataset_id,
                entity_id=resolved_entity,
                product_id=resolved_product,
                horizon=request.horizon,
                model_name="none",
                status=ForecastStatus.INSUFFICIENT_HISTORY.value,
                warnings=[f"Insufficient series observations: Series contains only {total_series_rows} rows; minimum 15 required for chronological splitting."],
                created_at=now,
            )

        # 3. Chronological Walk-Forward Train/Validation/Test Split
        split_config = DataProcessingConfig(
            train_ratio=effective_config.train_ratio,
            val_ratio=effective_config.val_ratio,
            test_ratio=1.0 - (effective_config.train_ratio + effective_config.val_ratio),
        )
        train_df, val_df, test_df, split_meta = self.splitter.split(
            series_df, mapping=mapping, config=split_config
        )

        train_rows = len(train_df)
        val_rows = len(val_df)
        if train_rows < 10 or val_rows < 3:
            return ForecastResult(
                forecast_id=forecast_id,
                dataset_id=request.dataset_id,
                entity_id=resolved_entity,
                product_id=resolved_product,
                horizon=request.horizon,
                model_name="none",
                status=ForecastStatus.INSUFFICIENT_HISTORY.value,
                warnings=[f"Partition rows too small (train={train_rows}, val={val_rows})."],
                created_at=now,
            )

        # 4. Evaluate Candidate Models on Validation Partition
        candidates = request.candidate_models or effective_config.candidate_models
        candidate_scores: List[ModelEvaluationScore] = []
        fitted_val_models: Dict[str, BaseForecastModel] = {}

        for cand_name in candidates:
            try:
                model_inst = self._instantiate_model(cand_name, effective_config)
                # Check history constraint per model
                req_rows = getattr(model_inst, "lookback", 5) + 5
                if train_rows < req_rows:
                    candidate_scores.append(
                        ModelEvaluationScore(
                            model_name=cand_name,
                            status=ForecastStatus.INSUFFICIENT_HISTORY.value,
                            error_message=f"Requires {req_rows} training rows, got {train_rows}.",
                        )
                    )
                    continue

                # Fit on train partition exclusively
                model_inst.fit(train_df, target_col=target_col, date_col=date_col)
                # Evaluate on validation partition
                metrics = model_inst.evaluate_validation(val_df, target_col=target_col, date_col=date_col)
                fitted_val_models[cand_name] = model_inst

                candidate_scores.append(
                    ModelEvaluationScore(
                        model_name=cand_name,
                        status=ForecastStatus.SUCCESS.value,
                        mae=metrics["MAE"],
                        rmse=metrics["RMSE"],
                        mape=metrics["MAPE"],
                        training_duration_seconds=model_inst.training_duration,
                    )
                )
            except Exception as exc:
                # Failure Isolation: individual model exceptions do NOT crash the agent
                candidate_scores.append(
                    ModelEvaluationScore(
                        model_name=cand_name,
                        status=ForecastStatus.FAILED.value,
                        error_message=str(exc),
                    )
                )

        # 5. Deterministic Model Ranking & Selection
        successful_scores = [s for s in candidate_scores if s.status == ForecastStatus.SUCCESS.value]
        if not successful_scores:
            return ForecastResult(
                forecast_id=forecast_id,
                dataset_id=request.dataset_id,
                entity_id=resolved_entity,
                product_id=resolved_product,
                horizon=request.horizon,
                model_name="none",
                status=ForecastStatus.NO_VALID_MODEL.value,
                warnings=[s.error_message or "Candidate failed" for s in candidate_scores],
                created_at=now,
            )

        metric_key = request.selection_metric.upper()
        if metric_key == "RMSE":
            successful_scores.sort(key=lambda s: s.rmse if s.rmse is not None else float("inf"))
        elif metric_key == "MAPE":
            successful_scores.sort(key=lambda s: s.mape if s.mape is not None else float("inf"))
        else:
            metric_key = "MAE"
            successful_scores.sort(key=lambda s: s.mae if s.mae is not None else float("inf"))

        winning_score = successful_scores[0]
        selected_model_name = winning_score.model_name
        ranking = [s.model_name for s in successful_scores]

        selection_reason = (
            f"Selected {selected_model_name.upper()} with lowest validation {metric_key} "
            f"({getattr(winning_score, metric_key.lower())}) among {len(successful_scores)} valid candidate(s)."
        )

        model_selection = ModelSelectionResult(
            candidate_scores=candidate_scores,
            ranking=ranking,
            selected_model=selected_model_name,
            selection_metric=metric_key,
            selection_reason=selection_reason,
        )

        # 6. Refit Winning Model on Combined (Train + Validation) Data
        # Strict Rule: Test partition is NEVER included in training or selection!
        history_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(by=date_col)
        refitted_winner = self._instantiate_model(selected_model_name, effective_config)
        refitted_winner.fit(history_df, target_col=target_col, date_col=date_col)

        # Preserve validation residuals from the validation round for prediction interval calibration
        val_winner = fitted_val_models[selected_model_name]
        refitted_winner.validation_residuals = val_winner.validation_residuals

        # 7. Generate Future Forecast Points
        # Generate contiguous future date sequence
        last_history_date = pd.to_datetime(history_df[date_col].iloc[-1])
        future_dates = [
            str((last_history_date + pd.Timedelta(days=i + 1)).date())
            for i in range(request.horizon)
        ]

        forecast_points = refitted_winner.predict(
            horizon=request.horizon,
            future_dates=future_dates,
            confidence_level=request.confidence_level,
        )

        # 8. Persist Trained Model Artifact
        saved_path = save_forecaster(
            refitted_winner,
            base_dir=self.persistence_dir,
            filename_prefix=f"{resolved_entity}_{resolved_product}",
        )

        # 9. Assemble Predictions List & Confidence Bounds
        preds_list: List[Dict[str, Any]] = [p.model_dump() for p in forecast_points]
        conf_intervals = {
            "lower_bounds": [p.lower_bound for p in forecast_points],
            "upper_bounds": [p.upper_bound for p in forecast_points],
            "confidence_level": request.confidence_level,
            "method": "native_prophet" if selected_model_name == "prophet" else "empirical_validation_residuals",
        }

        # 10. Assemble Result
        overall_status = (
            ForecastStatus.PARTIAL_SUCCESS.value
            if len(successful_scores) < len(candidates)
            else ForecastStatus.SUCCESS.value
        )

        meta = refitted_winner.get_model_metadata()

        return ForecastResult(
            forecast_id=forecast_id,
            dataset_id=request.dataset_id,
            entity_id=resolved_entity,
            product_id=resolved_product,
            horizon=request.horizon,
            model_name=selected_model_name,
            status=overall_status,
            predictions=forecast_points,
            forecast_points=forecast_points,
            confidence_intervals=conf_intervals,
            uncertainty_method=conf_intervals["method"],
            confidence_level=request.confidence_level,
            metrics={
                "validation_MAE": winning_score.mae or 0.0,
                "validation_RMSE": winning_score.rmse or 0.0,
                "validation_MAPE": winning_score.mape or 0.0,
            },
            model_selection=model_selection,
            model_metadata=meta,
            training_period={
                "start": str(pd.to_datetime(history_df[date_col].iloc[0]).date()),
                "end": str(pd.to_datetime(history_df[date_col].iloc[-1]).date()),
            },
            warnings=warnings,
            created_at=now,
        )
