"""Model Persistence and Artifact Storage for GlassBox-BI.

Saves and loads trained forecasting model checkpoints and metadata under models/saved/.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster


def save_forecaster(
    model: BaseForecastModel,
    base_dir: Union[str, Path] = "models/saved",
    filename_prefix: Optional[str] = None,
) -> Path:
    """Serializes a fitted forecast model to structured local storage.

    Args:
        model: Fitted BaseForecastModel instance.
        base_dir: Root persistence directory (defaults to models/saved).
        filename_prefix: Optional prefix for checkpoint name.

    Returns:
        Path to the saved checkpoint file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sub_dir = Path(base_dir) / model.model_name.lower()
    sub_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    if isinstance(model, ProphetForecaster):
        file_path = sub_dir / f"{prefix}{model.model_name}_{timestamp}.json"
    elif isinstance(model, LightGBMForecaster):
        file_path = sub_dir / f"{prefix}{model.model_name}_{timestamp}.joblib"
    elif isinstance(model, LSTMForecaster):
        file_path = sub_dir / f"{prefix}{model.model_name}_{timestamp}.pt"
    else:
        file_path = sub_dir / f"{prefix}{model.model_name}_{timestamp}.bin"

    model.save_model(file_path)
    return file_path


def load_forecaster(path: Union[str, Path], model_type: str) -> BaseForecastModel:
    """Loads a persisted model checkpoint given its filepath and architecture type.

    Args:
        path: Path to checkpoint file.
        model_type: Architecture type ('prophet', 'lightgbm', 'lstm').

    Returns:
        Deserialized BaseForecastModel instance.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path_obj}")

    mtype = model_type.lower()
    if mtype == "prophet":
        return ProphetForecaster.load_model(path_obj)
    elif mtype == "lightgbm":
        return LightGBMForecaster.load_model(path_obj)
    elif mtype == "lstm":
        return LSTMForecaster.load_model(path_obj)
    else:
        raise ValueError(f"Unsupported model type for deserialization: {model_type}")
