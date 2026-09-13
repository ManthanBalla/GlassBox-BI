"""GlassBox-BI Forecasting Module (Phase 4).

Architectural Responsibilities:
- Model-agnostic forecasting contract (BaseForecastModel)
- Candidate adapters: Prophet, LightGBM, LSTM
- GenericForecastingAgent orchestrating series filtering, train/validation split,
  model competition, dynamic ranking, refitting, and future prediction intervals
- Model checkpoint serialization and deserialization
"""

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.agent import GenericForecastingAgent
from backend.app.forecasting.persistence import save_forecaster, load_forecaster

__all__ = [
    "BaseForecastModel",
    "ProphetForecaster",
    "LightGBMForecaster",
    "LSTMForecaster",
    "GenericForecastingAgent",
    "save_forecaster",
    "load_forecaster",
]
