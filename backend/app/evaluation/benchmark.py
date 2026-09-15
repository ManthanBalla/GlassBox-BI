"""Forecasting Benchmark Engine for GlassBox-BI (Phase 5).

Coordinates rigorous, reproducible multi-model benchmark evaluations on the holdout test set:
1. Splits chronological series: Train (70%) -> Validation (15%) -> Test (15%)
2. Captures Phase 4 selection decision strictly via Validation performance
3. Fits candidate models on pre-test history (Train + Validation)
4. Evaluates candidate models on the holdout Test partition for the requested horizon
5. Computes formal test metrics (MAE, RMSE, MAPE) with zero-target protection
6. Enforces strict architectural separation: test results NEVER modify model selection.
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
from backend.app.schemas.forecasting import (
    ForecastingConfig,
    ForecastStatus,
)
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.schemas.evaluation import (
    BenchmarkResult,
    EvaluationRequest,
    ModelTestEvaluation,
)
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor
from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.evaluation.evaluator import FormalForecastEvaluator


class ForecastingBenchmarkEngine:
    """Universal benchmark engine executing rigorous test-set evaluations across forecasting models."""

    def __init__(
        self,
        default_config: Optional[ForecastingConfig] = None,
        splitter: Optional[TimeSeriesSplitter] = None,
        evaluator: Optional[FormalForecastEvaluator] = None,
    ) -> None:
        self.config = default_config or ForecastingConfig()
        self.splitter = splitter or TimeSeriesSplitter()
        self.evaluator = evaluator or FormalForecastEvaluator()

    def _instantiate_model(
        self,
        model_name: str,
        config: ForecastingConfig,
        random_seed: int = 42,
    ) -> BaseForecastModel:
        """Instantiates a clean candidate model adapter with deterministic seeding."""
        m_name = model_name.lower().strip()
        if m_name == "prophet":
            return ProphetForecaster(
                model_name="prophet",
                random_seed=random_seed,
                yearly_seasonality=config.prophet_yearly_seasonality,
                weekly_seasonality=config.prophet_weekly_seasonality,
                daily_seasonality=config.prophet_daily_seasonality,
                interval_width=config.confidence_level,
            )
        elif m_name == "lightgbm":
            return LightGBMForecaster(
                model_name="lightgbm",
                random_seed=random_seed,
                n_estimators=config.lgb_n_estimators,
                learning_rate=config.lgb_learning_rate,
                max_depth=config.lgb_max_depth,
                num_leaves=config.lgb_num_leaves,
            )
        elif m_name == "lstm":
            return LSTMForecaster(
                model_name="lstm",
                random_seed=random_seed,
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
            raise ValueError(f"Unsupported candidate forecasting model for evaluation: {model_name}")

    def _load_or_resolve_dataset(
        self,
        request: EvaluationRequest,
        provided_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, ColumnMapping]:
        """Resolves target dataset into a validated DataFrame and ColumnMapping."""
        df: Optional[pd.DataFrame] = provided_df

        if df is None:
            if request.dataset_path and Path(request.dataset_path).exists():
                df = pd.read_csv(request.dataset_path)
            elif Path("data/processed/synthetic/retail_processed.csv").exists():
                df = pd.read_csv("data/processed/synthetic/retail_processed.csv")
            elif Path("data/sample/retail_processed_sample.csv").exists():
                df = pd.read_csv("data/sample/retail_processed_sample.csv")
            elif Path("data/sample/retail_sample.csv").exists():
                raw_df = pd.read_csv("data/sample/retail_sample.csv")
                processor = GenericBusinessDataProcessor()
                df, _ = processor.process(raw_df, mapping=SYNTHETIC_RETAIL_MAPPING)
            else:
                raise FileNotFoundError(
                    "No valid dataset found for evaluation benchmark. "
                    "Please generate synthetic data or provide a valid dataset_path."
                )

        mapping = auto_detect_column_mapping(list(df.columns))
        return df, mapping

    def _generate_summary_table(
        self,
        evaluations: List[ModelTestEvaluation],
        phase4_selected_model: str,
        test_start: str,
        test_end: str,
        horizon: int,
        entity_id: str,
        product_id: str,
    ) -> str:
        """Constructs a clean, human-readable ASCII/Markdown comparison table."""
        header = (
            f"### FORECASTING BENCHMARK REPORT\n\n"
            f"- **Series**: {entity_id} / {product_id}\n"
            f"- **Test Holdout Period**: {test_start} -> {test_end} ({horizon} steps)\n"
            f"- **Phase 4 Validation Winner**: `{phase4_selected_model.upper()}` (Selected via Validation MAE)\n"
            f"- **Test Set Usage**: Independent Holdout (Strictly 0% used for model selection)\n\n"
            f"| Model | Test MAE | Test RMSE | Test MAPE | Val MAE | Phase 4 Selected? | Status |\n"
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        rows = []
        for ev in evaluations:
            mae_str = f"{ev.mae:.4f}" if ev.mae is not None else "N/A"
            rmse_str = f"{ev.rmse:.4f}" if ev.rmse is not None else "N/A"
            mape_str = f"{ev.mape:.2f}%" if ev.mape is not None else "N/A"
            val_mae_str = (
                f"{ev.validation_metrics.get('MAE', 0.0):.4f}"
                if ev.validation_metrics and "MAE" in ev.validation_metrics
                else "N/A"
            )
            selected_str = "⭐ **YES**" if ev.model_name == phase4_selected_model else "No"
            rows.append(
                f"| `{ev.model_name}` | {mae_str} | {rmse_str} | {mape_str} | {val_mae_str} | {selected_str} | {ev.status} |"
            )

        return header + "\n".join(rows) + "\n"

    def run_benchmark(
        self,
        request: EvaluationRequest,
        df: Optional[pd.DataFrame] = None,
        custom_config: Optional[ForecastingConfig] = None,
    ) -> BenchmarkResult:
        """Executes a formal, reproducible forecast benchmark evaluation on the holdout test set.

        Workflow:
        1. Resolves series observations and validates chronological history.
        2. Splits into Train (70%), Validation (15%), Test (15%) partitions.
        3. Simulates Phase 4 validation competition: fits candidates on Train, evaluates on Val.
           Records winning model based on Validation MAE (Test data is completely untouched).
        4. Fits candidates on pre-test history (Train + Val) up to the test set boundary.
        5. Evaluates each candidate independently on the holdout Test partition.
        6. Compares test predictions against actual test targets (MAE, RMSE, MAPE).
        7. Returns complete machine-readable BenchmarkResult preserving Phase 4 selection integrity.
        """
        now = datetime.now(timezone.utc)
        evaluation_id = f"eval_{now.strftime('%Y%m%d_%H%M%S')}"
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
                warnings.append(f"entity_id omitted; defaulting to first series entity '{resolved_entity}'.")
            series_df = series_df[series_df[entity_col].astype(str) == str(resolved_entity)]

        if product_col and product_col in series_df.columns:
            if not resolved_product:
                resolved_product = str(series_df[product_col].iloc[0])
                warnings.append(f"product_id omitted; defaulting to first product '{resolved_product}'.")
            series_df = series_df[series_df[product_col].astype(str) == str(resolved_product)]

        series_df = series_df.sort_values(by=date_col).reset_index(drop=True)
        total_series_rows = len(series_df)

        if total_series_rows < 15:
            raise ValueError(
                f"Insufficient series observations: Series has {total_series_rows} rows; "
                f"minimum 15 required for chronological splitting."
            )

        # 3. Chronological Walk-Forward Train/Validation/Test Split (70/15/15)
        split_config = DataProcessingConfig(
            train_ratio=effective_config.train_ratio,
            val_ratio=effective_config.val_ratio,
            test_ratio=1.0 - (effective_config.train_ratio + effective_config.val_ratio),
        )
        train_df, val_df, test_df, split_meta = self.splitter.split(
            series_df, mapping=mapping, config=split_config
        )

        test_rows = len(test_df)
        if test_rows < request.horizon:
            raise ValueError(
                f"Requested evaluation horizon ({request.horizon} days) exceeds available test set "
                f"observations ({test_rows} days from {split_meta.test_start_date} to {split_meta.test_end_date}). "
                f"Please choose a horizon <= {test_rows}."
            )

        # 4. Phase 4 Simulation: Candidate Model Selection strictly on Validation Partition
        candidates = request.candidate_models or ["prophet", "lightgbm", "lstm"]
        validation_scores: Dict[str, Dict[str, float]] = {}
        val_winner_name = "lightgbm"  # Default fallback
        best_val_mae = float("inf")

        for cand_name in candidates:
            try:
                cand_val_model = self._instantiate_model(
                    cand_name, effective_config, random_seed=request.random_seed
                )
                req_rows = getattr(cand_val_model, "lookback", 5) + 5
                if len(train_df) >= req_rows:
                    # Fit strictly on train_df
                    cand_val_model.fit(train_df, target_col=target_col, date_col=date_col)
                    # Evaluate strictly on val_df
                    val_metrics = cand_val_model.evaluate_validation(
                        val_df, target_col=target_col, date_col=date_col
                    )
                    validation_scores[cand_name] = val_metrics
                    if val_metrics["MAE"] < best_val_mae:
                        best_val_mae = val_metrics["MAE"]
                        val_winner_name = cand_name
            except Exception as exc:
                warnings.append(f"Validation phase candidate {cand_name} encountered error: {str(exc)}")

        # 5. Holdout Test Set Evaluation
        # Models are fitted on pre-test history (train + val) without peeking at test_df
        history_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(by=date_col)
        test_evaluations: List[ModelTestEvaluation] = []

        for cand_name in candidates:
            try:
                # Instantiate clean model for test evaluation
                eval_model = self._instantiate_model(
                    cand_name, effective_config, random_seed=request.random_seed
                )
                # Fit strictly on historical data prior to test boundary
                eval_model.fit(history_df, target_col=target_col, date_col=date_col)

                # Calibrate prediction intervals using validation residuals
                if cand_name in validation_scores:
                    eval_model.validation_residuals = np.array(
                        [validation_scores[cand_name].get("MAE", 1.0)]
                    )

                # Formally evaluate on the holdout test partition
                eval_result = self.evaluator.evaluate_model(
                    model=eval_model,
                    test_df=test_df,
                    horizon=request.horizon,
                    target_col=target_col,
                    date_col=date_col,
                    confidence_level=request.confidence_level,
                )

                # Attach Phase 4 validation metadata
                eval_result.is_validation_winner = (cand_name == val_winner_name)
                eval_result.validation_metrics = validation_scores.get(cand_name)
                test_evaluations.append(eval_result)

            except Exception as exc:
                test_evaluations.append(
                    ModelTestEvaluation(
                        model_name=cand_name,
                        status="FAILED",
                        error_message=str(exc),
                        horizon=request.horizon,
                        is_validation_winner=(cand_name == val_winner_name),
                        validation_metrics=validation_scores.get(cand_name),
                    )
                )

        # 6. Reporting-Only Test Winner (Academic Comparison, NOT Model Selection)
        successful_evals = [e for e in test_evaluations if e.status == "SUCCESS" and e.mae is not None]
        test_winner_reporting: Optional[str] = None
        if successful_evals:
            best_test_eval = min(successful_evals, key=lambda e: e.mae if e.mae is not None else float("inf"))
            test_winner_reporting = best_test_eval.model_name

        # 7. Format Benchmark Table
        summary_table = self._generate_summary_table(
            evaluations=test_evaluations,
            phase4_selected_model=val_winner_name,
            test_start=split_meta.test_start_date or "",
            test_end=str(pd.to_datetime(test_df[date_col].iloc[request.horizon - 1]).date()),
            horizon=request.horizon,
            entity_id=str(resolved_entity),
            product_id=str(resolved_product),
        )

        overall_status = "SUCCESS" if all(e.status == "SUCCESS" for e in test_evaluations) else "PARTIAL_SUCCESS"

        reproducibility = {
            "random_seed": request.random_seed,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "benchmark_timestamp": now.isoformat(),
        }

        return BenchmarkResult(
            evaluation_id=evaluation_id,
            dataset_id=request.dataset_id,
            entity_id=resolved_entity,
            product_id=resolved_product,
            status=overall_status,
            horizon=request.horizon,
            test_period={
                "start": split_meta.test_start_date,
                "end": str(pd.to_datetime(test_df[date_col].iloc[request.horizon - 1]).date()),
                "total_available_days": test_rows,
                "evaluated_days": request.horizon,
            },
            training_period={
                "train_start": split_meta.train_start_date,
                "train_end": split_meta.train_end_date,
                "val_start": split_meta.val_start_date,
                "val_end": split_meta.val_end_date,
                "history_end": split_meta.val_end_date,
            },
            data_split_info={
                "train_rows": split_meta.train_rows,
                "val_rows": split_meta.val_rows,
                "test_rows": split_meta.test_rows,
                "train_dates": f"{split_meta.train_start_date} -> {split_meta.train_end_date}",
                "val_dates": f"{split_meta.val_start_date} -> {split_meta.val_end_date}",
                "test_dates": f"{split_meta.test_start_date} -> {split_meta.test_end_date}",
            },
            models=test_evaluations,
            phase4_selected_model=val_winner_name,
            phase4_selection_metric="MAE",
            phase4_validation_score=round(best_val_mae, 4) if best_val_mae != float("inf") else None,
            selection_source="validation",
            test_set_used_for_selection=False,  # Strict Invariant: Always False
            test_winner_for_reporting_only=test_winner_reporting,
            benchmark_summary_markdown=summary_table,
            reproducibility=reproducibility,
            warnings=warnings,
            created_at=now,
        )
