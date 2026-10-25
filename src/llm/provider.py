"""Disabled-by-default provider boundary for OpenAI-compatible chat APIs."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
from typing import Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LLMProviderError(RuntimeError):
    """An LLM provider request failed or returned an invalid response."""


class LLMDisabledError(LLMProviderError):
    """LLM use was requested while the provider is disabled."""


class LLMProvider(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        """Generate a response from chat messages."""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and not _is_local_host(parsed.hostname):
            raise ValueError("Non-local LLM endpoints must use HTTPS")
        if not self.model.strip():
            raise ValueError("LLM model name cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout must be positive")


def _is_local_host(hostname: str) -> bool:
    host = hostname.lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleProvider:
    """Small synchronous adapter for OpenAI-compatible `/chat/completions` APIs."""

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        if not messages:
            raise ValueError("At least one chat message is required")
        normalized = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError("Each chat message requires a supported role and non-empty string content")
            normalized.append({"role": role, "content": content})

        payload: dict[str, object] = {"model": self.config.model, "messages": normalized}
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMProviderError(f"LLM provider returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise LLMProviderError(f"LLM provider connection failed: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMProviderError("LLM provider returned invalid JSON") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM provider response is missing assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM provider returned empty assistant content")
        return content


def create_llm_provider() -> LLMProvider | None:
    """Create the configured provider, or return None when LLM use is disabled."""
    provider_name = os.getenv("EQUIMASTER_LLM_PROVIDER", "disabled").strip().lower()
    if provider_name in ("", "disabled", "none"):
        return None
    if provider_name != "openai_compatible":
        raise ValueError(f"Unsupported EQUIMASTER_LLM_PROVIDER: {provider_name}")

    base_url = os.getenv("EQUIMASTER_LLM_BASE_URL", "").strip()
    model = os.getenv("EQUIMASTER_LLM_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError(
            "EQUIMASTER_LLM_BASE_URL and EQUIMASTER_LLM_MODEL are required "
            "for the openai_compatible provider"
        )
    timeout = float(os.getenv("EQUIMASTER_LLM_TIMEOUT_SECONDS", "30"))
    config = OpenAICompatibleConfig(
        base_url=base_url,
        model=model,
        api_key=os.getenv("EQUIMASTER_LLM_API_KEY") or None,
        timeout_seconds=timeout,
    )
    return OpenAICompatibleProvider(config)
