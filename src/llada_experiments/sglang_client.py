from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class GenerationResult:
    text: str
    raw: dict[str, Any]


class SGLangClient:
    """Small OpenAI-compatible client for SGLang experiments."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout_seconds: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        request_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        if request_id is not None:
            payload["rid"] = request_id

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return GenerationResult(text=text, raw=data)

    def completion(
        self,
        *,
        model: str,
        prompt: str,
        request_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        if request_id is not None:
            payload["rid"] = request_id

        response = httpx.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["text"]
        return GenerationResult(text=text, raw=data)
