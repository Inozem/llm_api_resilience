import os
from pathlib import Path
from typing import Optional

import pytest
from dotenv import load_dotenv
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY


load_dotenv(Path(__file__).parents[2] / ".env")

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _is_real_api_key(value: Optional[str]) -> bool:
    return bool(value and not value.startswith("__REPLACE_WITH_"))


def _latest_registry_model(provider: str) -> str:
    models = LLM_REGISTRY.providers[provider].models
    return next(iter(models))


@pytest.fixture(
    params=("openai", "anthropic", "google"),
    ids=("openai", "anthropic", "google"),
)
def configured_provider(request):
    provider = request.param
    api_key = os.getenv(_PROVIDER_ENV_KEYS[provider])
    if not _is_real_api_key(api_key):
        pytest.skip(
            f"{_PROVIDER_ENV_KEYS[provider]} is not configured; "
            "set it in .env to run this E2E case"
        )

    return {
        "name": provider,
        "model": _latest_registry_model(provider),
        "api_key": api_key,
    }
