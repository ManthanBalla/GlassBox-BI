"""LSTM Forecasting Adapter for GlassBox-BI (PyTorch).

Implements sequence-to-one recurrent neural network demand forecasting using PyTorch,
with train-only standard scaling, deterministic random seeding, early stopping,
and validation-residual-based prediction intervals.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.schemas.forecasting import ForecastPoint
from backend.app.forecasting.base import BaseForecastModel


class _PyTorchLSTMNetwork(nn.Module):
    """Lightweight sequence-to-one LSTM architecture."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last time step output: (batch_size, hidden_dim)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        prediction = self.fc(out)
        return prediction


class LSTMForecaster(BaseForecastModel):
    """PyTorch LSTM adapter implementing the universal BaseForecastModel interface."""

    def __init__(
        self,
        model_name: str = "lstm",
        random_seed: int = 42,
        lookback: int = 14,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 20,
        batch_size: int = 16,
        patience: int = 5,
    ) -> None:
        super().__init__(model_name=model_name, random_seed=random_seed)
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lookback_window = lookback
        self.uncertainty_method = "validation_residuals"

        self.net: Optional[_PyTorchLSTMNetwork] = None
        # Train-Only Scaler Parameters (Strict Zero-Leakage Guarantee)
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        self._historical_dates: List[pd.Timestamp] = []
        self._historical_targets: List[float] = []

    def _scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.scaler_mean) / (self.scaler_std + 1e-6)

    def _descale(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values * (self.scaler_std + 1e-6) + self.scaler_mean

    def _create_sequences(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.lookback):
            X.append(series[i : i + self.lookback])
            y.append(series[i + self.lookback])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "date",
        **kwargs: Any,
    ) -> "LSTMForecaster":
        """Fits LSTM neural network strictly on historical training observations."""
        start_t = time.perf_counter()
        self.target_col = target_col
        self.date_col = date_col
        self.training_rows = len(train_df)

        min_required = self.lookback + 5
        if self.training_rows < min_required:
            raise ValueError(
                f"LSTM with lookback={self.lookback} requires at least {min_required} training rows, got {self.training_rows}."
            )

        # 1. Deterministic Seeding
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        sorted_train = train_df.sort_values(by=date_col).reset_index(drop=True)
        self.training_start = str(pd.to_datetime(sorted_train[date_col].iloc[0]).date())
        self.training_end = str(pd.to_datetime(sorted_train[date_col].iloc[-1]).date())
        self._historical_dates = list(pd.to_datetime(sorted_train[date_col]))
        raw_targets = sorted_train[target_col].to_numpy(dtype=float)
        self._historical_targets = list(raw_targets)

        # 2. Fit Scaler ONLY on Training Target (Zero Leakage Rule)
        self.scaler_mean = float(np.mean(raw_targets))
        self.scaler_std = float(np.std(raw_targets)) if np.std(raw_targets) > 1e-6 else 1.0

        scaled_targets = self._scale(raw_targets)

        # 3. Create Sequential Training Pairs (X, y)
        X_train_np, y_train_np = self._create_sequences(scaled_targets)
        # Reshape X to (samples, seq_len, 1)
        X_train_t = torch.tensor(X_train_np).unsqueeze(-1)
        y_train_t = torch.tensor(y_train_np).unsqueeze(-1)

        # 4. DataLoader without Random Shuffling (Preserves Temporal Sequence)
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 5. Initialize Network and Optimizer
        self.net = _PyTorchLSTMNetwork(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.net.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # 6. Training Loop with Loss Convergence
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.net(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(dataset)

            # Simple early stopping
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.net.eval()
        self.is_fitted = True
        self.training_duration = time.perf_counter() - start_t
        self.feature_names = [f"lag_{i+1}" for i in range(self.lookback)]
        return self

    def predict(
        self,
        horizon: int,
        future_dates: Optional[List[str]] = None,
        confidence_level: float = 0.80,
        **kwargs: Any,
    ) -> List[ForecastPoint]:
        """Generates future predictions using iterative recursive sequence propagation."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"LSTMForecaster {self.model_name} is not fitted.")

        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {horizon}")

        # Resolve future dates
        if future_dates and len(future_dates) > 0:
            resolved_dates = [pd.to_datetime(d) for d in future_dates[:horizon]]
        else:
            last_date = self._historical_dates[-1]
            resolved_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

        # Prepare initial scaled lookback window from historical end
        current_window = self._scale(np.array(self._historical_targets[-self.lookback:], dtype=np.float32))

        forecast_points: List[ForecastPoint] = []
        self.net.eval()

        with torch.no_grad():
            for step_date in resolved_dates:
                # Shape: (1, lookback, 1)
                x_in = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                scaled_pred = float(self.net(x_in).item())

                # Descale and clamp non-negative
                raw_val = float(self._descale(scaled_pred))
                pred_val = max(0.0, round(raw_val, 4))

                # Slide window: append current scaled prediction, drop oldest
                current_window = np.append(current_window[1:], scaled_pred)

                # Compute prediction intervals via validation residuals
                low, high = self.compute_residual_prediction_interval(pred_val, confidence_level=confidence_level)

                forecast_points.append(
                    ForecastPoint(
                        date=str(step_date.date()),
                        prediction=pred_val,
                        lower_bound=low,
                        upper_bound=high,
                    )
                )

        return forecast_points

    def _get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }

    def _get_software_versions(self) -> Dict[str, str]:
        base_v = super()._get_software_versions()
        base_v["torch"] = torch.__version__
        return base_v

    def save_model(self, path: Union[str, Path]) -> None:
        """Serializes LSTM model state, scaler, and parameters to disk."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Cannot save unfitted LSTM model.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": self.model_name,
            "model_type": "LSTMForecaster",
            "hyperparameters": self._get_hyperparameters(),
            "target_col": self.target_col,
            "date_col": self.date_col,
            "training_rows": self.training_rows,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "validation_residuals": self.validation_residuals.tolist(),
            "historical_targets": self._historical_targets,
            "historical_dates": [str(d) for d in self._historical_dates],
            "state_dict": self.net.state_dict(),
        }
        torch.save(payload, path_obj)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "LSTMForecaster":
        """Deserializes LSTM model from disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path_obj}")

        payload = torch.load(path_obj, weights_only=False)
        hp = payload.get("hyperparameters", {})
        forecaster = cls(
            model_name=payload.get("model_name", "lstm"),
            lookback=hp.get("lookback", 14),
            hidden_dim=hp.get("hidden_dim", 32),
            num_layers=hp.get("num_layers", 1),
            dropout=hp.get("dropout", 0.1),
            learning_rate=hp.get("learning_rate", 0.01),
            epochs=hp.get("epochs", 20),
            batch_size=hp.get("batch_size", 16),
            patience=hp.get("patience", 5),
        )
        forecaster.target_col = payload.get("target_col", "target")
        forecaster.date_col = payload.get("date_col", "date")
        forecaster.training_rows = payload.get("training_rows", 0)
        forecaster.training_start = payload.get("training_start")
        forecaster.training_end = payload.get("training_end")
        forecaster.scaler_mean = payload.get("scaler_mean", 0.0)
        forecaster.scaler_std = payload.get("scaler_std", 1.0)
        forecaster.validation_residuals = np.array(payload.get("validation_residuals", []))
        forecaster._historical_targets = payload.get("historical_targets", [])
        forecaster._historical_dates = [pd.to_datetime(d) for d in payload.get("historical_dates", [])]

        forecaster.net = _PyTorchLSTMNetwork(
            input_dim=1,
            hidden_dim=forecaster.hidden_dim,
            num_layers=forecaster.num_layers,
            dropout=forecaster.dropout,
        )
        forecaster.net.load_state_dict(payload["state_dict"])
        forecaster.net.eval()
        forecaster.is_fitted = True
        return forecaster
