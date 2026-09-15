"""Feature Extraction and Background Sampling Adapter for Explainability (Phase 6).

Aligns tabular, sequence, and component data representations with the internal
states of candidate forecasting models (LightGBM, LSTM, Prophet), enforcing the
Zero-Leakage Principle: background reference distributions are sampled strictly
from the historical training partition.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster


class ExplainabilityFeatureAdapter:
    """Reconstructs exact model input vectors and extracts leakage-safe background samples."""

    @staticmethod
    def extract_prediction_features(
        model: BaseForecastModel,
        history_df: pd.DataFrame,
        prediction_date: Optional[str] = None,
        step_index: int = 0,
        target_col: str = "target",
        date_col: str = "date",
    ) -> Tuple[pd.DataFrame, float, str]:
        """Reconstructs the precise feature vector supplied to the model for a given forecast step.

        Returns:
            Tuple of (feature_row_df, model_point_prediction, resolved_date_str).
        """
        sorted_hist = history_df.sort_values(by=date_col).reset_index(drop=True)
        last_date = pd.to_datetime(sorted_hist[date_col].iloc[-1])

        # Resolve prediction date
        if prediction_date:
            target_dt = pd.to_datetime(prediction_date)
            # Find how many days after history
            days_ahead = (target_dt.date() - last_date.date()).days
            if days_ahead <= 0:
                step_idx = 0
                resolved_dt = last_date + pd.Timedelta(days=1)
            else:
                step_idx = max(0, days_ahead - 1)
                resolved_dt = target_dt
        else:
            step_idx = max(0, step_index)
            resolved_dt = last_date + pd.Timedelta(days=step_idx + 1)

        resolved_date_str = str(resolved_dt.date())

        if isinstance(model, LightGBMForecaster):
            return ExplainabilityFeatureAdapter._extract_lightgbm_features(
                model=model,
                sorted_hist=sorted_hist,
                step_idx=step_idx,
                step_dt=resolved_dt,
                resolved_date_str=resolved_date_str,
                target_col=target_col,
            )
        elif isinstance(model, LSTMForecaster):
            return ExplainabilityFeatureAdapter._extract_lstm_features(
                model=model,
                sorted_hist=sorted_hist,
                step_idx=step_idx,
                step_dt=resolved_dt,
                resolved_date_str=resolved_date_str,
                target_col=target_col,
            )
        elif isinstance(model, ProphetForecaster):
            return ExplainabilityFeatureAdapter._extract_prophet_features(
                model=model,
                step_dt=resolved_dt,
                resolved_date_str=resolved_date_str,
            )
        else:
            raise ValueError(f"Unsupported model type for feature adapter: {type(model)}")

    @staticmethod
    def _extract_lightgbm_features(
        model: LightGBMForecaster,
        sorted_hist: pd.DataFrame,
        step_idx: int,
        step_dt: pd.Timestamp,
        resolved_date_str: str,
        target_col: str,
    ) -> Tuple[pd.DataFrame, float, str]:
        """Reconstructs feature vector for LightGBM recursive multi-step forecasting."""
        extended_targets = list(model._historical_targets)
        # If step_idx > 0, recursively roll forward to step_idx
        if step_idx > 0:
            for s in range(step_idx):
                curr_dt = pd.to_datetime(model._historical_dates[-1]) + pd.Timedelta(days=s + 1)
                r_feat = ExplainabilityFeatureAdapter._build_lgb_row(
                    model=model,
                    extended_targets=extended_targets,
                    step_date=curr_dt,
                )
                pred = float(model.model.predict(r_feat)[0])  # type: ignore
                extended_targets.append(max(0.0, round(pred, 4)))

        # Build feature vector for the target step
        feature_row = ExplainabilityFeatureAdapter._build_lgb_row(
            model=model,
            extended_targets=extended_targets,
            step_date=step_dt,
        )

        pred_val = float(model.model.predict(feature_row)[0])  # type: ignore
        pred_val = max(0.0, round(pred_val, 4))
        return feature_row, pred_val, resolved_date_str

    @staticmethod
    def _build_lgb_row(
        model: LightGBMForecaster,
        extended_targets: List[float],
        step_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Helper to compute calendar, lag, and rolling features for LightGBM."""
        row_features = dict(model._last_known_features)

        # Calendar
        row_features["year"] = step_date.year
        row_features["month"] = step_date.month
        row_features["quarter"] = step_date.quarter
        row_features["day_of_week"] = step_date.dayofweek
        row_features["day_of_month"] = step_date.day
        row_features["day_of_year"] = step_date.dayofyear
        row_features["week_of_year"] = step_date.isocalendar().week
        row_features["is_weekend"] = 1 if step_date.dayofweek >= 5 else 0

        # Lags
        for lag in [1, 7, 14, 28]:
            lag_col = f"lag_{lag}"
            if lag_col in model.feature_names:
                if len(extended_targets) >= lag:
                    row_features[lag_col] = extended_targets[-lag]
                else:
                    row_features[lag_col] = extended_targets[0]

        # Rolling
        for w in [7, 14, 28]:
            mean_col = f"rolling_mean_{w}"
            std_col = f"rolling_std_{w}"
            if mean_col in model.feature_names:
                win_vals = extended_targets[-w:] if len(extended_targets) >= w else extended_targets
                row_features[mean_col] = float(np.mean(win_vals))
            if std_col in model.feature_names:
                win_vals = extended_targets[-w:] if len(extended_targets) >= w else extended_targets
                row_features[std_col] = float(np.std(win_vals)) if len(win_vals) > 1 else 0.0

        return pd.DataFrame([{col: row_features.get(col, 0.0) for col in model.feature_names}])

    @staticmethod
    def _extract_lstm_features(
        model: LSTMForecaster,
        sorted_hist: pd.DataFrame,
        step_idx: int,
        step_dt: pd.Timestamp,
        resolved_date_str: str,
        target_col: str,
    ) -> Tuple[pd.DataFrame, float, str]:
        """Extracts the univariate sequence window for LSTM."""
        import torch

        current_window = model._scale(np.array(model._historical_targets[-model.lookback:], dtype=np.float32))

        # Roll forward if step_idx > 0
        model.net.eval()  # type: ignore
        with torch.no_grad():
            for s in range(step_idx):
                x_in = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                scaled_pred = float(model.net(x_in).item())  # type: ignore
                current_window = np.append(current_window[1:], scaled_pred)

            x_target = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            final_scaled = float(model.net(x_target).item())  # type: ignore
            pred_val = max(0.0, round(float(model._descale(final_scaled)), 4))

        # Descaled window features (lag_1 to lag_{lookback})
        # lag_1 is the most recent observation in the window (index -1)
        # lag_k is the k-th most recent observation
        descaled_window = model._descale(current_window)
        feature_dict = {}
        for i in range(model.lookback):
            # lag_1 = most recent (descaled_window[-1])
            lag_num = i + 1
            feature_dict[f"lag_{lag_num}"] = float(descaled_window[-(i + 1)])

        feat_df = pd.DataFrame([feature_dict])
        return feat_df, pred_val, resolved_date_str

    @staticmethod
    def _extract_prophet_features(
        model: ProphetForecaster,
        step_dt: pd.Timestamp,
        resolved_date_str: str,
    ) -> Tuple[pd.DataFrame, float, str]:
        """Extracts the Prophet prediction DataFrame."""
        df_future = pd.DataFrame({"ds": [step_dt]})
        fcst = model.model.predict(df_future)  # type: ignore
        pred_val = max(0.0, round(float(fcst["yhat"].iloc[0]), 4))
        return df_future, pred_val, resolved_date_str

    @staticmethod
    def extract_background_sample(
        model: BaseForecastModel,
        train_df: pd.DataFrame,
        sample_size: int = 50,
        random_seed: int = 42,
        target_col: str = "target",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Samples a representative, leakage-safe background reference distribution from train_df.

        Zero-Leakage Principle: Reference distributions use historical training data exclusively.
        """
        rng = np.random.default_rng(seed=random_seed)

        if isinstance(model, LightGBMForecaster):
            feature_cols = [c for c in model.feature_names if c in train_df.columns]
            if not feature_cols:
                raise ValueError("No matching feature columns found in training DataFrame for LightGBM background.")

            avail_df = train_df[feature_cols].dropna()
            n_rows = len(avail_df)
            if n_rows == 0:
                avail_df = train_df[feature_cols].fillna(0.0)
                n_rows = len(avail_df)

            k = min(sample_size, n_rows)
            if k <= 0:
                raise ValueError("Training DataFrame contains zero valid rows for background sample.")

            indices = rng.choice(n_rows, size=k, replace=False if k <= n_rows else True)
            sampled = avail_df.iloc[indices].copy().reset_index(drop=True)
            return sampled

        elif isinstance(model, LSTMForecaster):
            # Create lookback sequence windows from train target
            targets = train_df[target_col].to_numpy(dtype=float)
            lookback = model.lookback
            if len(targets) < lookback + 1:
                raise ValueError(f"Training targets too short for LSTM lookback {lookback}.")

            windows = []
            for i in range(len(targets) - lookback):
                w = targets[i : i + lookback]
                # Map to lag features: lag_1 is w[-1], lag_lookback is w[0]
                row = {f"lag_{j + 1}": float(w[-(j + 1)]) for j in range(lookback)}
                windows.append(row)

            win_df = pd.DataFrame(windows)
            n_rows = len(win_df)
            k = min(sample_size, n_rows)
            indices = rng.choice(n_rows, size=k, replace=False if k <= n_rows else True)
            return win_df.iloc[indices].copy().reset_index(drop=True)

        elif isinstance(model, ProphetForecaster):
            # For Prophet, sample ds timestamps from training
            p_df = pd.DataFrame({"ds": pd.to_datetime(train_df[date_col])})
            n_rows = len(p_df)
            k = min(sample_size, n_rows)
            indices = rng.choice(n_rows, size=k, replace=False if k <= n_rows else True)
            return p_df.iloc[indices].copy().reset_index(drop=True)

        else:
            raise ValueError(f"Unsupported model type for background sampling: {type(model)}")
