from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.data import DemoMarketConfig, generate_demo_ohlcv
from quant_starter.factors import (
    analyze_composite,
    build_factor_history,
    calculate_factor_signals,
    factor_reference_catalog,
    mine_time_series_factors,
)
from quant_starter.news import (
    NewsArticle,
    NewsFeed,
    NewsProviderStatus,
    _deduplicate,
    _parse_rss_articles,
    _relevance_terms,
    _safe_http_url,
    fetch_online_news,
)
from quant_starter.realtime import (
    MarketSnapshot,
    RealtimeMarketData,
    fetch_realtime_market,
    intraday_records,
)
from web_app import (
    AgentResearchOptions,
    MarketService,
    ResearchJobManager,
    TTLCache,
    _merge_live_candle,
    _validate_agent_options,
    health,
    research_factors,
    research_providers,
    security_headers,
)
from quant_starter.agent_workflow import ProgressEvent


class FactorResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bars = generate_demo_ohlcv(
            "ALPHA",
            DemoMarketConfig(start="2021-01-01", end="2026-01-01", seed=41),
        )

    def test_factor_engine_returns_auditable_bounded_signals(self) -> None:
        signals = calculate_factor_signals(self.bars)
        self.assertEqual(len(signals), 21)
        self.assertAlmostEqual(sum(item.weight for item in signals), 1.0)
        self.assertTrue(all(-100 <= item.score <= 100 for item in signals))
        self.assertTrue(all(item.description for item in signals))
        self.assertEqual(
            {item.direction for item in signals}.difference(
                {"bullish", "bearish", "neutral"}
            ),
            set(),
        )
        referenced = [item for item in signals if item.reference_url]
        self.assertEqual(len(referenced), 8)
        self.assertTrue(all(item.formula for item in referenced))
        self.assertTrue(all(item.reference_url.startswith("https://") for item in referenced))

    def test_long_horizon_factor_is_explicit_when_history_is_short(self) -> None:
        signals = calculate_factor_signals(self.bars.tail(180))
        momentum = next(item for item in signals if item.key == "momentum_12_1")
        self.assertFalse(momentum.available)
        self.assertEqual(momentum.score, 0)
        self.assertEqual(momentum.history_required, 253)

    def test_composite_research_blends_five_strategies_and_risk_limits(self) -> None:
        result = analyze_composite(self.bars)
        self.assertEqual(len(result.strategies), 5)
        self.assertAlmostEqual(sum(item.weight for item in result.strategies), 1.0)
        self.assertIn(result.action, {"BUY", "HOLD", "SELL"})
        self.assertGreaterEqual(result.target_position, 0)
        self.assertLessEqual(result.target_position, 1)
        self.assertGreater(result.stop_loss_pct, 0)
        self.assertGreater(result.take_profit_pct, result.stop_loss_pct)
        self.assertIn(result.regime.key, {
            "high_volatility",
            "bull_trend",
            "bear_trend",
            "recovery",
            "range_bound",
        })

    def test_factor_mining_uses_rank_ic_without_scipy(self) -> None:
        results = mine_time_series_factors(self.bars, horizon_days=5)
        self.assertEqual(len(results), 14)
        self.assertTrue(all(item.observations >= 40 for item in results))
        self.assertTrue(
            all(-1 <= item.information_coefficient <= 1 for item in results)
        )
        self.assertGreaterEqual(results[0].observations, results[-1].observations - 30)
        self.assertIn("amihud_liquidity", {item.key for item in results})
        self.assertIn("momentum_12_1", {item.key for item in results})

    def test_factor_history_is_bounded_and_chronological(self) -> None:
        history = build_factor_history(self.bars, periods=30)
        self.assertEqual(len(history), 30)
        self.assertEqual([row["date"] for row in history], sorted(row["date"] for row in history))
        self.assertTrue(all(-100 <= row["trend"] <= 100 for row in history))
        self.assertEqual(
            set(history[-1]),
            {"date", "trend", "momentum", "reversal", "risk", "liquidity", "flow", "session"},
        )

    def test_factor_analysis_rejects_short_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "80"):
            calculate_factor_signals(self.bars.head(40))


class OnlineNewsTests(unittest.TestCase):
    @staticmethod
    def _article(provider: str, title: str, url: str, published: str) -> NewsArticle:
        return NewsArticle(
            article_id=f"{provider}-{title}",
            title=title,
            publisher="Test Wire",
            published_at=published,
            url=url,
            provider=provider,
            provider_label=provider,
            summary="测试摘要",
        )

    def test_external_news_links_allow_only_public_http_urls(self) -> None:
        self.assertEqual(_safe_http_url("javascript:alert(1)"), "")
        self.assertEqual(_safe_http_url("http://127.0.0.1/private"), "")
        self.assertEqual(_safe_http_url("https://example.com/story#tracking"), "https://example.com/story")
        self.assertEqual(
            _safe_http_url("http://www.cninfo.com.cn/a?time=2026-07-14 18:16:27"),
            "http://www.cninfo.com.cn/a?time=2026-07-14%2018:16:27",
        )

    def test_rss_parser_normalizes_link_time_and_summary(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><item>
        <title>Quarterly result</title>
        <link>https://example.com/report</link>
        <pubDate>Tue, 14 Jul 2026 12:30:00 +0000</pubDate>
        <description><![CDATA[<b>Revenue grew</b> year over year.]]></description>
        </item></channel></rss>"""
        items = _parse_rss_articles(
            payload,
            provider="test-rss",
            provider_label="Test RSS",
            limit=5,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].publisher, "example.com")
        self.assertEqual(items[0].summary, "Revenue grew year over year.")
        self.assertEqual(items[0].published_at, "2026-07-14T12:30:00+00:00")

    def test_news_aggregator_deduplicates_and_isolates_provider_failure(self) -> None:
        duplicate = self._article(
            "yahoo-finance",
            "NVIDIA announces platform",
            "https://example.com/nvda",
            "2026-07-15T01:00:00+00:00",
        )
        with (
            mock.patch(
                "quant_starter.news._fetch_yahoo_news",
                return_value=[duplicate, duplicate],
            ),
            mock.patch(
                "quant_starter.news._fetch_nasdaq_news",
                side_effect=ConnectionError("provider offline"),
            ),
            mock.patch(
                "quant_starter.news._fetch_sec_filings",
                return_value=[],
            ),
        ):
            feed = fetch_online_news(
                "nasdaq",
                "NVDA",
                company_name="NVIDIA",
                timeout_seconds=2,
            )
        self.assertEqual(len(feed.items), 1)
        self.assertEqual(
            {item.provider: item.status for item in feed.providers},
            {
                "yahoo-finance": "ok",
                "nasdaq-rss": "error",
                "sec-edgar": "empty",
            },
        )
        self.assertIn("Nasdaq", feed.warnings[0])

    def test_news_ranking_prioritizes_relevance_then_official_sources(self) -> None:
        generic = self._article(
            "yahoo-finance",
            "Broad market update",
            "https://example.com/market",
            "2026-07-15T03:00:00+00:00",
        )
        relevant = self._article(
            "yahoo-finance",
            "NVIDIA launches a new platform",
            "https://example.com/nvidia",
            "2026-07-15T01:00:00+00:00",
        )
        filing = NewsArticle(
            article_id="sec-filing",
            title="8-K filing",
            publisher="SEC",
            published_at="2026-07-15T02:00:00+00:00",
            url="https://www.sec.gov/filing",
            provider="sec-edgar",
            provider_label="SEC EDGAR",
            source_kind="filing",
        )
        ranked = _deduplicate(
            [generic, filing, relevant],
            3,
            relevance_terms=_relevance_terms("NVDA", "NVIDIA"),
        )
        self.assertEqual(
            [item.article_id for item in ranked],
            [relevant.article_id, filing.article_id, generic.article_id],
        )


class RealtimeMarketTests(unittest.TestCase):
    @staticmethod
    def _tencent_quote_body() -> bytes:
        fields = [""] * 80
        fields[0] = "51"
        fields[1] = "宁德时代"
        fields[2] = "300750"
        fields[3] = "364.01"
        fields[4] = "359.06"
        fields[5] = "352.60"
        fields[6] = "436316"
        fields[30] = "20260714151409"
        fields[33] = "364.63"
        fields[34] = "352.49"
        fields[35] = "364.01/436316/15701875682"
        return ('v_sz300750="' + "~".join(fields) + '";').encode("gb18030")

    @staticmethod
    def _tencent_minute_body() -> bytes:
        payload = {
            "code": 0,
            "msg": "",
            "data": {
                "sz300750": {
                    "data": {
                        "date": "20260714",
                        "data": [
                            "0930 352.60 100 3526000.00",
                            "0931 353.20 160 5645200.00",
                            "0932 354.00 250 8831200.00",
                        ],
                    }
                }
            },
        }
        return json.dumps(payload).encode("utf-8")

    def test_tencent_realtime_quote_and_minute_data_are_normalized(self) -> None:
        def response(request, **_kwargs):
            if "minute/query" in request.full_url:
                return self._tencent_minute_body()
            return self._tencent_quote_body()

        with mock.patch("quant_starter.realtime._read_http_bytes", side_effect=response):
            live = fetch_realtime_market("a-share", "300750")
        self.assertEqual(live.snapshot.name, "宁德时代")
        self.assertAlmostEqual(live.snapshot.change_pct, 4.95 / 359.06, places=6)
        self.assertEqual(live.snapshot.volume, 43_631_600)
        self.assertEqual(len(live.intraday), 3)
        self.assertEqual(float(live.intraday.iloc[1]["Volume"]), 6_000)
        self.assertAlmostEqual(
            float(live.intraday.iloc[0]["AveragePrice"]), 352.60, places=2
        )

    def test_yahoo_realtime_data_are_normalized(self) -> None:
        timestamps = [
            int(datetime(2026, 7, 14, 13, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 14, 13, 31, tzinfo=timezone.utc).timestamp()),
        ]
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": "NVDA",
                            "longName": "NVIDIA Corporation",
                            "currency": "USD",
                            "exchangeTimezoneName": "America/New_York",
                            "regularMarketTime": timestamps[-1],
                            "regularMarketPrice": 211.0,
                            "previousClose": 208.0,
                            "regularMarketDayHigh": 212.0,
                            "regularMarketDayLow": 207.0,
                            "regularMarketVolume": 1_500,
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [208.5, 210.0],
                                    "high": [210.5, 211.5],
                                    "low": [208.0, 209.5],
                                    "close": [210.0, 211.0],
                                    "volume": [1_000, 500],
                                }
                            ]
                        },
                    }
                ],
            }
        }
        with mock.patch(
            "quant_starter.realtime._read_http_bytes",
            return_value=json.dumps(payload).encode("utf-8"),
        ):
            live = fetch_realtime_market("nasdaq", "NVDA")
        self.assertEqual(live.snapshot.symbol, "NVDA")
        self.assertEqual(live.snapshot.price, 211.0)
        self.assertEqual(live.snapshot.change, 3.0)
        self.assertEqual(float(live.intraday.iloc[-1]["CumulativeVolume"]), 1_500)

    def test_nasdaq_public_chart_is_used_when_yahoo_is_blocked(self) -> None:
        timestamps = [
            int(datetime(2026, 7, 14, 13, 30, tzinfo=timezone.utc).timestamp() * 1000),
            int(datetime(2026, 7, 14, 20, 0, tzinfo=timezone.utc).timestamp() * 1000),
        ]
        payload = {
            "data": {
                "symbol": "NVDA",
                "company": "NVIDIA Corporation Common Stock",
                "lastSalePrice": "$211.80",
                "netChange": "+8.27",
                "percentageChange": "+4.06%",
                "previousClose": "$203.53",
                "volume": "124,381,661",
                "chart": [
                    {"x": timestamps[0], "y": 208.2},
                    {"x": timestamps[1], "y": 211.8},
                ],
            },
            "status": {"rCode": 200},
        }
        with (
            mock.patch(
                "quant_starter.realtime._yahoo_intraday_result",
                side_effect=ConnectionError("Yahoo blocked"),
            ),
            mock.patch(
                "quant_starter.realtime._read_http_bytes",
                return_value=json.dumps(payload).encode("utf-8"),
            ),
        ):
            live = fetch_realtime_market("nasdaq", "NVDA")
        self.assertEqual(live.snapshot.provider, "nasdaq-public-chart")
        self.assertEqual(live.snapshot.volume, 124_381_661)
        self.assertEqual(live.snapshot.price, 211.8)
        self.assertEqual(len(live.intraday), 2)
        self.assertEqual(float(live.intraday["Volume"].sum()), 0.0)

    def test_intraday_records_respect_limit(self) -> None:
        with mock.patch(
            "quant_starter.realtime._read_http_bytes",
            side_effect=lambda request, **_kwargs: self._tencent_minute_body()
            if "minute/query" in request.full_url
            else self._tencent_quote_body(),
        ):
            live = fetch_realtime_market("a-share", "300750")
        records = intraday_records(live.intraday, limit=2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["price"], 354.0)


class WebServiceTests(unittest.TestCase):
    def test_factor_registry_exposes_external_methodology(self) -> None:
        payload = research_factors()
        self.assertEqual(payload["total_factors"], 21)
        self.assertEqual(payload["referenced_factors"], 8)
        self.assertEqual(payload["factors"], factor_reference_catalog())

    def test_research_provider_registry_exposes_supported_services(self) -> None:
        payload = research_providers()
        providers = {item["id"]: item for item in payload["providers"]}
        self.assertEqual(set(providers), {"openai", "deepseek", "qwen", "ollama"})
        self.assertFalse(providers["ollama"]["requires_api_key"])
        self.assertTrue(all("api_key" not in item for item in payload["providers"]))

    def test_market_service_caches_news_and_supports_forced_refresh(self) -> None:
        article = OnlineNewsTests._article(
            "nasdaq-rss",
            "NVIDIA update",
            "https://www.nasdaq.com/articles/nvidia-update",
            "2026-07-15T01:00:00+00:00",
        )
        feed = NewsFeed(
            market="nasdaq",
            symbol="NVDA",
            query="NVIDIA NVDA stock",
            items=(article,),
            providers=(
                NewsProviderStatus("nasdaq-rss", "Nasdaq 官方 RSS", "ok", 1),
            ),
            warnings=(),
            fetched_at="2026-07-15T01:01:00+00:00",
        )
        market_service = MarketService()
        with mock.patch("web_app.fetch_online_news", return_value=feed) as loader:
            first = market_service.get_news("nasdaq", "NVDA", limit=12)
            second = market_service.get_news("nasdaq", "NVDA", limit=12)
            forced = market_service.get_news("nasdaq", "NVDA", limit=12, force=True)
        self.assertEqual(first, second)
        self.assertEqual(forced["items"][0]["title"], "NVIDIA update")
        self.assertEqual(loader.call_count, 2)

    def test_online_news_replaces_stale_evidence_warning(self) -> None:
        bars = generate_demo_ohlcv("ALPHA").tail(300)
        bars.attrs["provider"] = "test-bars"
        market_service = MarketService()
        online_feed = {
            "items": [
                {
                    "title": "Company update",
                    "publisher": "Official Wire",
                    "published_at": "2026-07-15T01:00:00+00:00",
                    "url": "https://example.com/update",
                }
            ],
            "warnings": [],
        }
        with (
            mock.patch.object(market_service, "get_bars", return_value=bars),
            mock.patch.object(market_service, "get_news", return_value=online_feed),
            mock.patch(
                "web_app.fetch_research_evidence",
                return_value=(
                    {},
                    [],
                    ["A股基本面/新闻暂不可用：fundamental source offline"],
                ),
            ),
        ):
            result = market_service.run_agents(
                "a-share",
                "300750",
                options=AgentResearchOptions(fetch_details=True),
            )
        self.assertEqual(result["evidence"]["news_items"], 1)
        self.assertTrue(
            any("基本面暂不可用" in warning for warning in result["warnings"])
        )
        self.assertFalse(any("基本面/新闻" in warning for warning in result["warnings"]))

    def test_agent_options_reject_empty_analyst_team_before_queueing(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少选择"):
            _validate_agent_options(AgentResearchOptions(selected_analysts=()))

    def test_online_market_service_calls_all_twelve_agents(self) -> None:
        requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                decision = json.dumps(
                    {
                        "action": "HOLD",
                        "confidence": 72,
                        "target_allocation": 0.35,
                        "stop_loss_pct": 0.08,
                        "take_profit_pct": 0.16,
                        "time_horizon": "20 个交易日",
                        "rationale": "模拟在线智能体已完成可审计研判。",
                    },
                    ensure_ascii=False,
                )
                payload = {
                    "choices": [{"message": {"content": decision}}],
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        bars = generate_demo_ohlcv(
            "ALPHA",
            DemoMarketConfig(start="2024-01-01", end="2026-01-01", seed=91),
        )
        bars.attrs["provider"] = "test-online-bars"
        progress: list[ProgressEvent] = []
        options = AgentResearchOptions(
            mode="online",
            provider="openai",
            model="test-model",
            api_key="test-secret-key",
            timeout_seconds=10,
        )
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "RAVENWATCHAGENTS_OPENAI_BASE_URL": (
                        f"http://127.0.0.1:{server.server_address[1]}/v1"
                    )
                },
            ):
                market_service = MarketService()
                with mock.patch.object(market_service, "get_bars", return_value=bars):
                    result = market_service.run_agents(
                        "nasdaq",
                        "NVDA",
                        options=options,
                        progress=progress.append,
                    )
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

        self.assertEqual(len(requests), 12)
        self.assertTrue(all(item["path"] == "/v1/chat/completions" for item in requests))
        self.assertTrue(all(item["authorization"] == "Bearer test-secret-key" for item in requests))
        self.assertTrue(all(item["body"]["model"] == "test-model" for item in requests))
        self.assertEqual(result["agent_summary"]["status_counts"], {"completed": 12})
        self.assertTrue(result["agent_summary"]["all_completed"])
        self.assertEqual(len([item for item in progress if item.status == "completed"]), 12)
        self.assertNotIn("test-secret-key", json.dumps(result, ensure_ascii=False))

    def test_research_job_manager_publishes_progress_and_result(self) -> None:
        manager = ResearchJobManager(max_active=1)

        def runner(progress, _cancel_event):
            progress(ProgressEvent(1, 1, "market", "技术分析师", "completed", "已完成"))
            return {"agent_summary": {"total": 1}, "decision": {"action": "HOLD"}}

        job = manager.create(runner)
        deadline = time.monotonic() + 2
        while job.status not in manager.TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = job.snapshot()
        events, _ = job.events_after(0)
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["result"]["decision"]["action"], "HOLD")
        self.assertEqual([item["event"] for item in events], ["state", "progress", "state"])

    def test_desktop_token_protects_api_routes(self) -> None:
        from fastapi.responses import JSONResponse
        from starlette.requests import Request

        async def call_next(_request):
            return JSONResponse({"status": "ok"})

        def request(query: bytes = b"") -> Request:
            return Request(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": "/api/health",
                    "raw_path": b"/api/health",
                    "query_string": query,
                    "headers": [],
                    "client": ("test", 1234),
                    "server": ("test", 80),
                    "root_path": "",
                }
            )

        with mock.patch.dict(os.environ, {"RAVENWATCHAGENTS_DESKTOP_TOKEN": "secret"}):
            denied = asyncio.run(security_headers(request(), call_next))
            accepted = asyncio.run(
                security_headers(request(b"desktop_token=secret"), call_next)
            )
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(accepted.status_code, 200)

    def test_live_candle_converts_a_share_volume_to_lots(self) -> None:
        bars = generate_demo_ohlcv("ALPHA").tail(100)
        timestamp = pd.Timestamp("2026-07-14 09:30", tz="Asia/Shanghai")
        intraday = pd.DataFrame(
            {
                "Price": [100.0, 102.0],
                "AveragePrice": [100.0, 101.0],
                "Volume": [10_000.0, 20_000.0],
                "CumulativeVolume": [10_000.0, 30_000.0],
                "Amount": [1_000_000.0, 2_040_000.0],
            },
            index=[timestamp, timestamp + pd.offsets.Minute(1)],
        )
        snapshot = MarketSnapshot(
            market="a-share",
            symbol="300750",
            name="宁德时代",
            currency="CNY",
            price=102,
            previous_close=99,
            change=3,
            change_pct=3 / 99,
            open=100,
            high=102,
            low=100,
            volume=30_000,
            amount=3_040_000,
            timestamp=timestamp.isoformat(),
            session_status="open",
            provider="test",
        )
        merged = _merge_live_candle(
            bars, RealtimeMarketData(snapshot=snapshot, intraday=intraday)
        )
        self.assertEqual(float(merged.iloc[-1]["Volume"]), 300)
        self.assertEqual(float(merged.iloc[-1]["Close"]), 102)

    def test_ttl_cache_returns_stale_value_when_refresh_fails(self) -> None:
        cache = TTLCache()
        key = ("test",)
        value = cache.get_or_load(
            key, lambda: "fresh", ttl_seconds=0, stale_seconds=60
        )
        self.assertEqual(value, "fresh")
        stale = cache.get_or_load(
            key,
            lambda: (_ for _ in ()).throw(ConnectionError("offline")),
            ttl_seconds=0,
            stale_seconds=60,
        )
        self.assertEqual(stale, "fresh")

    def test_health_declares_realtime_markets(self) -> None:
        payload = health()
        self.assertTrue(payload["realtime"])
        self.assertEqual(
            payload["markets"], ["a-share", "nasdaq", "hk", "global"]
        )
        self.assertTrue(payload["instrument_search"])


if __name__ == "__main__":
    unittest.main()
