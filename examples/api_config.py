"""Shared configuration helpers for live API examples."""

import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without requiring extra dependencies."""

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def load_workspace_env() -> None:
    """Load the repository .env while preserving CI environment variables.

    Examples can live either in the workspace-level ``example_scripts``
    directory or in the package-level ``llm_api_resilience/examples``
    directory, so support both layouts.
    """

    examples_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        examples_root / ".env",
        examples_root / "llm_api_resilience" / ".env",
    ]

    for env_path in candidate_paths:
        if env_path.is_file():
            _load_env_file(env_path)
            return


def required_api_key(env_name: str) -> str:
    """Return a configured API key without ever printing its value."""

    value = os.getenv(env_name)
    if not value or value.startswith("__REPLACE_WITH_"):
        raise RuntimeError(f"Set {env_name} in llm_api_resilience/.env first")
    return value
