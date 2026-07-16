from __future__ import annotations

import asyncio
from email.message import Message
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from urllib import error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.llm_client import (
    LLMRequestError,
    LLMSettings,
    OpenAICompatibleClient,
    provider_profile,
)
from web_app import ProviderConnectionRequest, test_research_provider


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.payload


def _success_response(content: str = "CONNECTED") -> _Response:
    return _Response(
        {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": "private reasoning must not be returned",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 31,
                "completion_tokens": 9,
                "total_tokens": 40,
                "prompt_cache_hit_tokens": 12,
                "prompt_cache_miss_tokens": 19,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }
    )


class DeepSeekClientTests(unittest.TestCase):
    def settings(self, **overrides) -> LLMSettings:
        values = {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key": "test-deepseek-key",
            "temperature": 0.2,
            "timeout_seconds": 30,
            "provider_id": "deepseek",
            "thinking_mode": "enabled",
            "reasoning_effort": "high",
            "max_tokens": 1200,
            "max_retries": 2,
        }
        values.update(overrides)
        return LLMSettings(**values)

    def test_profile_exposes_current_v4_defaults(self) -> None:
        profile = provider_profile("deepseek")
        payload = profile.public_dict(
            server_key_configured=False,
            server_model="",
        )
        self.assertEqual(profile.base_url, "https://api.deepseek.com")
        self.assertEqual(profile.default_model, "deepseek-v4-flash")
        self.assertEqual(
            profile.recommended_models,
            ("deepseek-v4-flash", "deepseek-v4-pro"),
        )
        self.assertTrue(payload["supports_thinking"])
        self.assertNotIn("api_key", payload)

    def test_deepseek_payload_uses_thinking_and_records_usage(self) -> None:
        client = OpenAICompatibleClient(self.settings())
        with mock.patch(
            "quant_starter.llm_client.request.urlopen",
            return_value=_success_response("final answer"),
        ) as urlopen:
            content = client.complete("system", "user")

        sent_request = urlopen.call_args.args[0]
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(sent_request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer test-deepseek-key")
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertEqual(content, "final answer")
        self.assertNotIn("private reasoning", content)
        self.assertEqual(
            client.usage_summary(),
            {
                "requests": 1,
                "attempts": 1,
                "prompt_tokens": 31,
                "completion_tokens": 9,
                "total_tokens": 40,
                "reasoning_tokens": 7,
                "cache_hit_tokens": 12,
                "cache_miss_tokens": 19,
            },
        )

    def test_disabled_thinking_omits_reasoning_effort(self) -> None:
        client = OpenAICompatibleClient(self.settings(thinking_mode="disabled"))
        with mock.patch(
            "quant_starter.llm_client.request.urlopen",
            return_value=_success_response(),
        ) as urlopen:
            client.complete("system", "user")
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)

    def test_rate_limit_retries_then_succeeds(self) -> None:
        headers = Message()
        headers["Retry-After"] = "0.1"
        rate_limit = error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            429,
            "rate limited",
            headers,
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        client = OpenAICompatibleClient(self.settings())
        with (
            mock.patch(
                "quant_starter.llm_client.request.urlopen",
                side_effect=[rate_limit, _success_response()],
            ) as urlopen,
            mock.patch("quant_starter.llm_client.time.sleep") as sleep,
        ):
            self.assertEqual(client.complete("system", "user"), "CONNECTED")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(client.usage_summary()["attempts"], 2)

    def test_invalid_key_has_actionable_message_and_no_secret(self) -> None:
        unauthorized = error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            401,
            "unauthorized",
            Message(),
            io.BytesIO(
                b'{"error":{"message":"invalid test-deepseek-key"}}'
            ),
        )
        client = OpenAICompatibleClient(self.settings(max_retries=0))
        with mock.patch(
            "quant_starter.llm_client.request.urlopen",
            side_effect=unauthorized,
        ):
            with self.assertRaisesRegex(LLMRequestError, "API Key") as caught:
                client.complete("system", "user")
        self.assertNotIn("test-deepseek-key", str(caught.exception))

    def test_connection_diagnostic_does_not_echo_key(self) -> None:
        fake_client = mock.Mock()
        fake_client.complete.return_value = "CONNECTED"
        fake_client.usage_summary.return_value = {"requests": 1, "total_tokens": 8}
        runtime = {
            "mode": "online",
            "provider": "deepseek",
            "provider_label": "DeepSeek V4",
            "model": "deepseek-v4-flash",
            "server_key_used": False,
        }
        request_payload = ProviderConnectionRequest(
            provider="deepseek",
            api_key="diagnostic-secret",
        )
        with mock.patch(
            "web_app._resolve_llm_runtime",
            return_value=(fake_client, runtime),
        ):
            result = asyncio.run(test_research_provider(request_payload))
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["llm"]["usage"]["requests"], 1)
        self.assertNotIn("diagnostic-secret", serialized)


if __name__ == "__main__":
    unittest.main()
