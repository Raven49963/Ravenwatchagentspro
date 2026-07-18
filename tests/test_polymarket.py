from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_starter.polymarket import (
    GAMMA_SEARCH_URL,
    PolymarketSnapshot,
    assess_polymarket,
    fetch_polymarket_snapshot,
)


NOW = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


def _event(
    *,
    event_id: str,
    title: str,
    question: str,
    probability: float,
    best_bid: float | None = None,
    best_ask: float | None = None,
) -> dict[str, object]:
    market: dict[str, object] = {
        "id": f"market-{event_id}",
        "question": question,
        "slug": f"market-{event_id}",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps([str(probability), str(1 - probability)]),
        "lastTradePrice": str(probability),
        "volumeNum": 1_500_000,
        "volume24hr": 125_000,
        "liquidityNum": 180_000,
        "endDate": "2027-12-31T00:00:00Z",
        "active": True,
        "closed": False,
        "updatedAt": "2026-07-19T07:58:00Z",
        "resolutionSource": "Official public announcement",
    }
    if best_bid is not None:
        market["bestBid"] = best_bid
    if best_ask is not None:
        market["bestAsk"] = best_ask
    return {
        "id": f"event-{event_id}",
        "title": title,
        "slug": f"event-{event_id}",
        "active": True,
        "closed": False,
        "markets": [market],
    }


class FakeSearch:
    def __init__(self, responses: dict[str, list[dict[str, object]]]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str, timeout_seconds: float) -> dict[str, object]:
        self.urls.append(url)
        self.assert_request(url, timeout_seconds)
        query = parse_qs(urlparse(url).query)["q"][0]
        return {"events": self.responses.get(query, [])}

    @staticmethod
    def assert_request(url: str, timeout_seconds: float) -> None:
        assert url.startswith(GAMMA_SEARCH_URL + "?")
        assert timeout_seconds >= 2
        params = parse_qs(urlparse(url).query)
        assert params["events_status"] == ["active"]
        assert params["keep_closed_markets"] == ["0"]


class PolymarketParsingTests(unittest.TestCase):
    def test_direct_market_uses_tight_orderbook_midpoint(self) -> None:
        search = FakeSearch(
            {
                "Nvidia": [
                    _event(
                        event_id="nvda-cap",
                        title="Will Nvidia surpass a $5 trillion market cap?",
                        question="Will Nvidia surpass a $5 trillion market cap in 2027?",
                        probability=0.68,
                        best_bid=0.70,
                        best_ask=0.74,
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                company_name="NVIDIA Corporation",
                cache_dir=Path(directory),
                now=NOW,
                fetch_json=search,
            )

        self.assertEqual(snapshot.provider_status, "ok")
        self.assertEqual(snapshot.source_mode, "live")
        self.assertEqual(len(snapshot.items), 1)
        item = snapshot.items[0]
        self.assertAlmostEqual(item.yes_probability, 0.72)
        self.assertEqual(item.probability_source, "orderbook-midpoint")
        self.assertEqual(item.relevance_kind, "direct")
        self.assertEqual(item.impact_sign, 1)
        self.assertAlmostEqual(item.directional_score, 44.0)
        self.assertGreater(item.quality_score, 70)
        self.assertTrue(all(url.startswith(GAMMA_SEARCH_URL) for url in search.urls))

        assessment = assess_polymarket(snapshot)
        self.assertTrue(assessment["available"])
        self.assertGreater(assessment["directional_score"], 0)
        self.assertEqual(assessment["included_market_count"], 1)
        self.assertEqual(len(assessment["process"]), 5)

    def test_macro_recession_market_maps_event_probability_to_bearish(self) -> None:
        search = FakeSearch(
            {
                "US recession": [
                    _event(
                        event_id="recession",
                        title="Will the US enter a recession?",
                        question="Will the US enter a recession before 2027?",
                        probability=0.75,
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = fetch_polymarket_snapshot(
                "nasdaq",
                "AMD",
                company_name="Advanced Micro Devices Inc.",
                cache_dir=Path(directory),
                now=NOW,
                fetch_json=search,
            )

        recession = snapshot.items[0]
        self.assertEqual(recession.relevance_kind, "macro")
        self.assertEqual(recession.impact_sign, -1)
        self.assertAlmostEqual(recession.directional_score, -50.0)
        self.assertLess(assess_polymarket(snapshot)["directional_score"], 0)

    def test_ambiguous_event_is_visible_but_excluded_from_score(self) -> None:
        search = FakeSearch(
            {
                "Nvidia": [
                    _event(
                        event_id="release",
                        title="Will Nvidia release a new GPU in 2027?",
                        question="Will Nvidia release a new GPU in 2027?",
                        probability=0.81,
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=Path(directory),
                now=NOW,
                fetch_json=search,
            )

        assessment = assess_polymarket(snapshot)
        self.assertEqual(assessment["market_count"], 1)
        self.assertEqual(assessment["included_market_count"], 0)
        self.assertFalse(assessment["available"])
        self.assertFalse(assessment["markets"][0]["included"])
        self.assertIn("仅作背景展示", assessment["markets"][0]["exclusion_reason"])


class PolymarketCacheTests(unittest.TestCase):
    def test_live_snapshot_round_trips_provider_status_and_falls_back(self) -> None:
        search = FakeSearch(
            {
                "Nvidia": [
                    _event(
                        event_id="nvda-cap",
                        title="Will Nvidia surpass a $5 trillion market cap?",
                        question="Will Nvidia surpass a $5 trillion market cap in 2027?",
                        probability=0.65,
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            live = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=cache_dir,
                now=NOW,
                fetch_json=search,
            )
            cache_payload = json.loads((cache_dir / "nasdaq-NVDA.json").read_text("utf-8"))
            restored = PolymarketSnapshot.from_dict(cache_payload["snapshot"])

            def fail(_url: str, _timeout: float) -> dict[str, object]:
                raise OSError("network unavailable")

            fallback = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=cache_dir,
                now=NOW + timedelta(hours=2),
                force=True,
                fetch_json=fail,
            )

        self.assertEqual(live.provider_status, "ok")
        self.assertEqual(restored.provider_status, "ok")
        self.assertEqual(fallback.provider_status, "cached")
        self.assertEqual(fallback.source_mode, "cache-stale")
        self.assertTrue(fallback.stale)
        self.assertGreaterEqual(fallback.cache_age_seconds, 7_200)
        self.assertIn("已回退本地快照", " ".join(fallback.warnings))

    def test_offline_mode_uses_cache_without_network(self) -> None:
        search = FakeSearch(
            {
                "Nvidia": [
                    _event(
                        event_id="nvda-cap",
                        title="Will Nvidia surpass a $5 trillion market cap?",
                        question="Will Nvidia surpass a $5 trillion market cap in 2027?",
                        probability=0.65,
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=cache_dir,
                now=NOW,
                fetch_json=search,
            )

            def unexpected(_url: str, _timeout: float) -> dict[str, object]:
                raise AssertionError("offline mode must not call the network")

            cached = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=cache_dir,
                now=NOW + timedelta(days=1),
                force=True,
                offline=True,
                fetch_json=unexpected,
            )

        self.assertEqual(cached.source_mode, "cache-offline")
        self.assertTrue(cached.stale)
        self.assertEqual(len(cached.items), 1)

    def test_corrupt_offline_cache_returns_auditable_unavailable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "nasdaq-NVDA.json").write_text("not-json", encoding="utf-8")
            snapshot = fetch_polymarket_snapshot(
                "nasdaq",
                "NVDA",
                cache_dir=cache_dir,
                now=NOW,
                offline=True,
            )

        self.assertEqual(snapshot.source_mode, "offline-empty")
        self.assertFalse(snapshot.items)
        self.assertIn("缓存不可读", " ".join(snapshot.warnings))
        self.assertFalse(assess_polymarket(snapshot)["available"])


if __name__ == "__main__":
    unittest.main()
