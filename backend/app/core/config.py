import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    """Application baseline configuration settings."""
    app_name: str = os.getenv("APP_NAME", "GlassBox-BI")
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    version: str = "0.1.0-alpha"

settings = Settings()
