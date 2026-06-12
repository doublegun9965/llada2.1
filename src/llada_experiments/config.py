from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    output_dir: Path


def load_settings(env_file: str | os.PathLike[str] | None = ".env") -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Settings(
        base_url=os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1").rstrip("/"),
        api_key=os.getenv("SGLANG_API_KEY", "EMPTY"),
        model=os.getenv("SGLANG_MODEL", "llada2.1"),
        timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
    )
