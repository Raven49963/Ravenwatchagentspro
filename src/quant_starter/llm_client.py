from __future__ import annotations

from dataclasses import dataclass
import json
from urllib import error, parse, request
import uuid


class LLMRequestError(RuntimeError):
    """Raised when an OpenAI-compatible request cannot be completed."""


@dataclass(frozen=True)
class LLMProviderProfile:
    provider_id: str
    label: str
    base_url: str
    api_key_env: str
    requires_api_key: bool = True

    def public_dict(self, *, server_key_configured: bool, server_model: str) -> dict:
        return {
            "id": self.provider_id,
            "label": self.label,
            "base_url": self.base_url,
            "requires_api_key": self.requires_api_key,
            "server_key_configured": server_key_configured,
            "server_model": server_model,
        }


LLM_PROVIDER_PROFILES = {
    "openai": LLMProviderProfile(
        "openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY"
    ),
    "deepseek": LLMProviderProfile(
        "deepseek", "DeepSeek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
    ),
    "qwen": LLMProviderProfile(
        "qwen",
        "Qwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    "ollama": LLMProviderProfile(
        "ollama",
        "Ollama",
        "http://127.0.0.1:11434/v1",
        "",
        requires_api_key=False,
    ),
}


def provider_profile(provider_id: str) -> LLMProviderProfile:
    normalized = provider_id.strip().lower()
    try:
        return LLM_PROVIDER_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"不支持的在线模型服务：{provider_id}") from exc


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.2
    timeout_seconds: int = 120

    def validate(self) -> None:
        parsed = parse.urlparse(self.base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM 地址必须是有效的 http:// 或 https:// URL。")
        if not self.model.strip():
            raise ValueError("在线智能体模式必须填写模型名称。")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("模型 temperature 必须在 0 到 2 之间。")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM 超时时间必须大于 0。")


class OpenAICompatibleClient:
    """Small dependency-free client for /v1/chat/completions endpoints."""

    def __init__(self, settings: LLMSettings) -> None:
        settings.validate()
        self.settings = settings

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TradingAgentsPro/1.6",
            "X-Client-Request-Id": str(uuid.uuid4()),
        }
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"

        http_request = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(
                http_request, timeout=self.settings.timeout_seconds
            ) as response:
                body_bytes = response.read(2_000_001)
                if len(body_bytes) > 2_000_000:
                    raise LLMRequestError("LLM 返回内容超过 2 MB 限制。")
                body = body_bytes.decode("utf-8")
        except error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
            except Exception:
                detail = ""
            suffix = f"：{detail}" if detail else ""
            raise LLMRequestError(f"LLM 接口返回 HTTP {exc.code}{suffix}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LLMRequestError(f"无法连接 LLM 接口：{exc}") from exc

        try:
            decoded = json.loads(body)
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMRequestError("LLM 返回格式不符合 chat/completions 规范。") from exc
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("text")
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("LLM 返回了空内容。")
        return content.strip()
