"""Application settings and environment configuration."""

import os
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application baseline configuration settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="GlassBox-BI", description="Application service name")
    app_env: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=True, description="Enable debug mode")
    version: str = Field(default="0.2.0-alpha", description="API Version")
    backend_host: str = Field(default="127.0.0.1", description="Backend listening host")
    backend_port: int = Field(default=8000, description="Backend listening port")
    secret_key: str = Field(default="development-secret-key-glassbox-bi", description="Secret key for auth/sessions")
    allowed_origins: Union[str, List[str]] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
