from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any
from urllib import error, parse, request
import uuid

from .metadata import PRODUCT_USER_AGENT


class LLMRequestError(RuntimeError):
    """Raised when an OpenAI-compatible request cannot be completed."""


@dataclass(frozen=True)
class LLMProviderProfile:
    provider_id: str
    label: str
    base_url: str
    api_key_env: str
    requires_api_key: bool = True
    default_model: str = ""
    recommended_models: tuple[str, ...] = ()
    docs_url: str = ""
    supports_thinking: bool = False

    def public_dict(self, *, server_key_configured: bool, server_model: str) -> dict:
        return {
            "id": self.provider_id,
            "label": self.label,
            "base_url": self.base_url,
            "requires_api_key": self.requires_api_key,
            "server_key_configured": server_key_configured,
            "server_model": server_model,
            "default_model": self.default_model,
            "recommended_models": list(self.recommended_models),
            "docs_url": self.docs_url,
            "supports_thinking": self.supports_thinking,
        }


LLM_PROVIDER_PROFILES = {
    "deepseek": LLMProviderProfile(
        provider_id="deepseek",
        label="DeepSeek V4",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
        recommended_models=("deepseek-v4-flash", "deepseek-v4-pro"),
        docs_url="https://api-docs.deepseek.com/",
        supports_thinking=True,
    ),
    "openai": LLMProviderProfile(
        provider_id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "qwen": LLMProviderProfile(
        provider_id="qwen",
        label="Qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "ollama": LLMProviderProfile(
        provider_id="ollama",
        label="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",
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
    provider_id: str = "custom"
    thinking_mode: str = "enabled"
    reasoning_effort: str = "high"
    max_tokens: int = 2_000
    max_retries: int = 2

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
        if self.thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("思考模式必须是 enabled 或 disabled。")
        if self.reasoning_effort not in {"high", "max"}:
            raise ValueError("推理强度必须是 high 或 max。")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("重试次数必须在 0 到 5 之间。")


class OpenAICompatibleClient:
    """Dependency-free client for OpenAI-compatible chat completions APIs."""

    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    MAX_RESPONSE_BYTES = 2_000_000

    def __init__(self, settings: LLMSettings) -> None:
        settings.validate()
        self.settings = settings
        self._usage_lock = threading.Lock()
        self._usage = {
            "requests": 0,
            "attempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
        }
        if self.settings.provider_id == "deepseek":
            payload["max_tokens"] = self.settings.max_tokens
            payload["thinking"] = {"type": self.settings.thinking_mode}
            if self.settings.thinking_mode == "enabled":
                payload["reasoning_effort"] = self.settings.reasoning_effort

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": PRODUCT_USER_AGENT,
            "X-Client-Request-Id": str(uuid.uuid4()),
        }
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"

        encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.settings.max_retries + 1):
            http_request = request.Request(
                endpoint,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            self._increment_attempts()
            try:
                with request.urlopen(
                    http_request, timeout=self.settings.timeout_seconds
                ) as response:
                    body_bytes = response.read(self.MAX_RESPONSE_BYTES + 1)
                    if len(body_bytes) > self.MAX_RESPONSE_BYTES:
                        raise LLMRequestError("LLM 返回内容超过 2 MB 限制。")
                    body = body_bytes.decode("utf-8")
                break
            except error.HTTPError as exc:
                if (
                    exc.code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.settings.max_retries
                ):
                    retry_after = (
                        exc.headers.get("Retry-After") if exc.headers is not None else None
                    )
                    time.sleep(self._retry_delay(attempt, retry_after))
                    continue
                raise self._http_error(exc) from exc
            except (error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.settings.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                provider = self._provider_label()
                raise LLMRequestError(
                    f"无法连接 {provider} API，请检查网络、代理和接口地址。"
                ) from exc
        else:  # pragma: no cover - loop always returns or raises
            raise LLMRequestError("LLM 请求未能完成。")

        try:
            decoded = json.loads(body)
            message = decoded["choices"][0]["message"]
            content = message["content"]
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
        self._record_usage(decoded.get("usage"))
        return content.strip()

    def usage_summary(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def _increment_attempts(self) -> None:
        with self._usage_lock:
            self._usage["attempts"] += 1

    def _record_usage(self, usage: Any) -> None:
        usage = usage if isinstance(usage, dict) else {}
        completion_details = usage.get("completion_tokens_details")
        completion_details = (
            completion_details if isinstance(completion_details, dict) else {}
        )
        values = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
            "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
            "cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0),
        }
        with self._usage_lock:
            self._usage["requests"] += 1
            for key, value in values.items():
                if isinstance(value, int) and value >= 0:
                    self._usage[key] += value

    def _provider_label(self) -> str:
        try:
            return provider_profile(self.settings.provider_id).label
        except ValueError:
            return "LLM"

    def _http_error(self, exc: error.HTTPError) -> LLMRequestError:
        detail = self._response_error_detail(exc)
        if self.settings.provider_id == "deepseek":
            messages = {
                400: "DeepSeek 请求格式无效，请检查模型和推理参数。",
                401: "DeepSeek API Key 无效，请检查后重试。",
                402: "DeepSeek 账户余额不足，请检查计费状态。",
                422: "DeepSeek 请求参数不被接受，请检查模型 ID 与推理设置。",
                429: "DeepSeek 请求过于频繁或并发过高，请稍后重试。",
                500: "DeepSeek 服务内部错误，请稍后重试。",
                503: "DeepSeek 服务繁忙，请稍后重试。",
            }
            base_message = messages.get(exc.code, f"DeepSeek API 返回 HTTP {exc.code}。")
        else:
            base_message = f"{self._provider_label()} API 返回 HTTP {exc.code}。"
        if detail and exc.code not in {401, 402}:
            return LLMRequestError(f"{base_message} 服务信息：{detail}")
        return LLMRequestError(base_message)

    def _response_error_detail(self, exc: error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            return ""
        try:
            decoded = json.loads(raw)
            error_payload = decoded.get("error", decoded)
            if isinstance(error_payload, dict):
                detail = str(error_payload.get("message", ""))
            else:
                detail = str(error_payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            detail = raw
        api_key = self.settings.api_key.strip()
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        return " ".join(detail.split())[:240]

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(5.0, max(0.1, float(retry_after)))
            except ValueError:
                pass
        return min(2.0, 0.35 * (2**attempt))
