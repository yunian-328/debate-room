from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    custom_models: str = ""
    mock_mode: bool = True
    database_path: Path = PROJECT_ROOT / "data" / "debate_room.db"
    scenario_config: Path = PROJECT_ROOT / "config" / "scenarios.yaml"
    request_timeout_seconds: float = 30.0
    max_retries_per_model: int = 1
    callback_timeout_seconds: float = 10.0
    github_webhook_secret: str = ""
    github_token: str = ""
    preview_api_token: str = ""
    fact_search_provider: str = "wikimedia"
    brave_search_api_key: str = ""
    wikimedia_contact_url: str = ""
    app_name: str = "Debate Room · 多模型内容质检"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            custom_models=os.getenv("CUSTOM_MODELS", ""),
            mock_mode=_as_bool(os.getenv("OPENROUTER_MOCK"), True),
            database_path=Path(
                os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "debate_room.db"))
            ),
            scenario_config=Path(
                os.getenv(
                    "SCENARIO_CONFIG", str(PROJECT_ROOT / "config" / "scenarios.yaml")
                )
            ),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            max_retries_per_model=int(os.getenv("MAX_RETRIES_PER_MODEL", "1")),
            callback_timeout_seconds=float(os.getenv("CALLBACK_TIMEOUT_SECONDS", "10")),
            github_webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            preview_api_token=os.getenv("DEBATE_ROOM_API_TOKEN", ""),
            fact_search_provider=(os.getenv("FACT_SEARCH_PROVIDER", "").strip().lower() or "wikimedia"),
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
            wikimedia_contact_url=os.getenv("WIKIMEDIA_CONTACT_URL", "").strip(),
        )
