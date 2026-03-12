"""
Application configuration.

Loads settings from configs/app.yaml with environment variable overrides.
All sensitive values (API keys) come from environment variables only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve project root regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict if it doesn't exist."""
    if path.exists():
        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    Central settings object.

    Values are resolved in this priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. configs/app.yaml
    4. Field defaults (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "TripGenie"
    app_version: str = "0.1.0"
    tripgenie_env: str = Field(default="development", alias="TRIPGENIE_ENV")
    debug: bool = Field(default=False, alias="TRIPGENIE_DEBUG")
    log_level: str = Field(default="INFO", alias="TRIPGENIE_LOG_LEVEL")

    # --- LLM ---
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    # --- API server ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        alias="CORS_ORIGINS",
    )

    # --- Feature flags ---
    enable_eval_endpoint: bool = Field(default=True, alias="ENABLE_EVAL_ENDPOINT")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")

    # --- Derived from YAML (set via _load_from_yaml) ---
    ranking_top_k: int = 5
    ranking_feature_weights: dict[str, float] = Field(
        default={
            "budget_match": 0.25,
            "location_score": 0.20,
            "review_score": 0.18,
            "transport_match": 0.10,
            "museum_affinity": 0.08,
            "romantic_match": 0.07,
            "family_match": 0.05,
            "nightlife_match": 0.04,
            "business_match": 0.03,
        }
    )
    retrieval_max_candidates: int = 20
    workflow_step_timeout: int = 20

    # --- Data paths ---
    hotels_path: Path = _PROJECT_ROOT / "data" / "hotels.csv"
    attractions_path: Path = _PROJECT_ROOT / "data" / "attractions.csv"
    restaurants_path: Path = _PROJECT_ROOT / "data" / "restaurants.csv"
    city_guides_path: Path = _PROJECT_ROOT / "data" / "city_guides.json"
    sample_queries_path: Path = _PROJECT_ROOT / "data" / "sample_user_queries.json"
    prompts_path: Path = _PROJECT_ROOT / "configs" / "prompts.yaml"

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"mock", "openai", "openai_compatible"}
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        return v.upper()

    @property
    def is_mock_llm(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def model_post_init(self, __context: Any) -> None:
        """Overlay YAML config values onto settings that weren't set via env."""
        yaml_cfg = _load_yaml_config(_PROJECT_ROOT / "configs" / "app.yaml")

        ranking_cfg = yaml_cfg.get("ranking", {})
        if "top_k" in ranking_cfg and not os.environ.get("RANKING_TOP_K"):
            self.ranking_top_k = ranking_cfg["top_k"]
        if "feature_weights" in ranking_cfg:
            self.ranking_feature_weights = ranking_cfg["feature_weights"]

        retrieval_cfg = yaml_cfg.get("retrieval", {})
        if "max_candidate_hotels" in retrieval_cfg:
            self.retrieval_max_candidates = retrieval_cfg["max_candidate_hotels"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()
