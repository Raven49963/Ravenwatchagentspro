from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.local_evidence import (
    assess_fundamentals,
    assess_news,
    build_local_evidence,
)
from quant_starter.research_data import (
    _nasdaq_official_fundamentals,
    fetch_fundamental_snapshot,
)
from web_app import QuantValidationRequest


class FundamentalScoringTests(unittest.TestCase):
    def test_available_fields_are_scored_and_missing_fields_are_excluded(self) -> None:
        fields = {
            "displayName": "Example Corp",
            "peRatio": 18,
            "yieldPercent": 2.5,
            "revenueGrowth": 0.18,
            "earningsGrowth": 0.22,
            "profitMargins": 0.16,
            "returnOnEquity": 0.19,
            "debtToEquity": 55,
            "debtRatio": 0.36,
            "currentRatio": 1.8,
            "reportDate": "2026-03-31",
        }
        snapshot = {
            "symbol": "TEST",
            "fields": fields,
            "field_sources": {key: "Test Source" for key in fields},
            "providers": [{"status": "ok"}],
            "warnings": [],
        }
        result = assess_fundamentals(snapshot)
        self.assertTrue(result["available"])
        self.assertEqual(result["available_metrics"], 9)
        self.assertEqual(result["total_metrics"], 16)
        self.assertGreater(result["coverage"], 0.5)
        self.assertGreater(result["confidence"], 60)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertAlmostEqual(
            result["score"], 50 + result["directional_score"] / 2, places=2
        )
        price_to_book = next(
            metric for metric in result["metrics"] if metric["key"] == "price_to_book"
        )
        self.assertFalse(price_to_book["available"])
        self.assertIsNone(price_to_book["score"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_provider_failure_preserves_partial_a_share_snapshot(self) -> None:
        with (
            mock.patch(
                "quant_starter.research_data._msn_fundamentals",
                return_value={"displayName": "Test A", "peRatio": 20, "_source_url": "https://example.com/a"},
            ),
            mock.patch(
                "quant_starter.research_data._a_share_financials",
                side_effect=ConnectionError("financial source offline"),
            ),
            mock.patch(
                "quant_starter.research_data._eastmoney_financial_analysis",
                return_value={},
            ),
        ):
            snapshot = fetch_fundamental_snapshot(
                "a-share", "300750", timeout_seconds=2
            ).to_dict()
        self.assertEqual(snapshot["fields"]["peRatio"], 20)
        statuses = {item["provider"]: item["status"] for item in snapshot["providers"]}
        self.assertEqual(statuses["msn-finance"], "ok")
        self.assertEqual(statuses["sina-financials"], "error")
        self.assertTrue(any("financial source offline" in item for item in snapshot["warnings"]))

    def test_official_value_wins_while_cross_source_conflict_is_preserved(self) -> None:
        with (
            mock.patch(
                "quant_starter.research_data._msn_fundamentals",
                return_value={"peRatio": 20.0, "_source_url": "https://msn.example/value"},
            ),
            mock.patch(
                "quant_starter.research_data._yfinance_fundamentals",
                return_value={"profitMargins": 0.30, "_source_url": "https://yahoo.example/value"},
            ),
            mock.patch(
                "quant_starter.research_data._eastmoney_financial_analysis",
                return_value={"profitMargins": 0.10, "reportDate": "2026-03-31"},
            ),
            mock.patch(
                "quant_starter.research_data._nasdaq_official_fundamentals",
                return_value={"profitMargins": 0.20, "reportDate": "2026-03-31"},
            ),
            mock.patch(
                "quant_starter.research_data._sec_company_fundamentals",
                return_value={"profitMargins": 0.21, "reportDate": "2026-03-31"},
            ),
        ):
            snapshot = fetch_fundamental_snapshot(
                "nasdaq", "TEST", timeout_seconds=2
            ).to_dict()
        self.assertAlmostEqual(snapshot["fields"]["profitMargins"], 0.21)
        self.assertEqual(snapshot["field_sources"]["profitMargins"], "SEC XBRL Company Facts")
        self.assertEqual(len(snapshot["field_evidence"]["profitMargins"]), 4)
        metric_quality = snapshot["quality"]["metric_quality"]["profit_margin"]
        self.assertEqual(metric_quality["source_count"], 4)
        self.assertLess(metric_quality["agreement"], 0.8)
        self.assertTrue(any("跨源偏差" in warning for warning in snapshot["warnings"]))

    def test_nasdaq_official_financials_are_normalized(self) -> None:
        payload = {
            "data": {
                "incomeStatementTable": {
                    "headers": {"value2": "1/25/2026", "value3": "1/26/2025"},
                    "rows": [
                        {"value1": "Total Revenue", "value2": "$200,000", "value3": "$100,000"},
                        {"value1": "Net Income", "value2": "$40,000", "value3": "$20,000"},
                    ],
                },
                "balanceSheetTable": {
                    "headers": {"value2": "1/25/2026"},
                    "rows": [
                        {"value1": "Total Assets", "value2": "$500,000"},
                        {"value1": "Total Liabilities", "value2": "$200,000"},
                        {"value1": "Total Equity", "value2": "$300,000"},
                        {"value1": "Total Current Assets", "value2": "$180,000"},
                        {"value1": "Total Current Liabilities", "value2": "$90,000"},
                        {"value1": "Long-Term Debt", "value2": "$30,000"},
                    ],
                },
            }
        }
        with mock.patch(
            "quant_starter.research_data._read_bytes",
            return_value=json.dumps(payload).encode("utf-8"),
        ):
            fields = _nasdaq_official_fundamentals("TEST")
        self.assertEqual(fields["reportDate"], "2026-01-25")
        self.assertAlmostEqual(fields["revenueGrowth"], 1.0)
        self.assertAlmostEqual(fields["profitMargins"], 0.2)
        self.assertAlmostEqual(fields["debtRatio"], 0.4)
        self.assertAlmostEqual(fields["currentRatio"], 2.0)


class NewsAndCompositeEvidenceTests(unittest.TestCase):
    def test_news_confidence_rewards_verified_sources_without_inflating_events(self) -> None:
        base_item = {
            "title": "Company raises earnings guidance",
            "summary": "profit growth",
            "published_at": "2026-07-15T12:00:00+00:00",
            "source_kind": "media",
            "credibility": "财经媒体",
        }
        single = assess_news(
            {
                "items": [{**base_item, "verification_count": 1, "verification_score": 25}],
                "providers": [{"status": "ok"}],
            },
            now=datetime(2026, 7, 16, tzinfo=timezone.utc),
        )
        verified = assess_news(
            {
                "items": [
                    {
                        **base_item,
                        "verification_count": 5,
                        "verification_score": 88,
                        "verification_status": "five-source",
                    }
                ],
                "providers": [{"status": "ok"}],
            },
            now=datetime(2026, 7, 16, tzinfo=timezone.utc),
        )
        self.assertEqual(single["article_count"], verified["article_count"])
        self.assertGreater(verified["confidence"], single["confidence"])
        self.assertEqual(verified["five_source_verified_count"], 1)

    def test_recent_official_event_outweighs_old_media_event(self) -> None:
        feed = {
            "items": [
                {
                    "title": "公司获批并上调盈利预期",
                    "summary": "业务增长",
                    "published_at": "2026-07-15T12:00:00+00:00",
                    "source_kind": "filing",
                    "credibility": "监管机构官方",
                },
                {
                    "title": "Company faces lawsuit and downgrade",
                    "summary": "risk warning",
                    "published_at": "2026-05-01T12:00:00+00:00",
                    "source_kind": "media",
                    "credibility": "财经媒体",
                },
            ],
            "providers": [{"status": "ok"}, {"status": "ok"}],
            "warnings": [],
        }
        result = assess_news(
            feed,
            now=datetime(2026, 7, 16, tzinfo=timezone.utc),
        )
        self.assertGreater(result["directional_score"], 20)
        self.assertGreater(result["score"], 50)
        self.assertAlmostEqual(
            result["score"], 50 + result["directional_score"] / 2, places=2
        )
        self.assertEqual(result["positive_count"], 1)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["catalysts"][0]["title"], "公司获批并上调盈利预期")

    def test_composite_renormalizes_missing_components(self) -> None:
        result = build_local_evidence(
            technical_score=25,
            technical_confidence=60,
            quant_validation={
                "available": True,
                "latest_score": -0.1,
                "robustness_score": 70,
                "folds": [{}, {}, {}],
                "verdict": "较稳健",
            },
            fundamentals={"available": False},
            news={"available": False},
        )
        available = [item for item in result["components"] if item["available"]]
        self.assertAlmostEqual(sum(item["effective_weight"] for item in available), 1.0, places=5)
        self.assertAlmostEqual(result["coverage"], 0.65)
        self.assertEqual(set(result["missing_components"]), {"基本面", "新闻事件"})
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertAlmostEqual(
            result["score"], 50 + result["directional_score"] / 2, places=2
        )
        self.assertTrue(
            all(0 <= item["score"] <= 100 for item in available)
        )
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _component_payloads(score: float, confidence: int = 90) -> tuple[dict, dict]:
        fundamentals = {
            "available": True,
            "score": score,
            "confidence": confidence,
            "available_metrics": 12,
            "total_metrics": 16,
        }
        news = {
            "available": True,
            "score": score,
            "confidence": confidence,
            "article_count": 8,
            "official_ratio": 0.5,
        }
        return fundamentals, news

    def test_bearish_evidence_uses_nonnegative_rating_and_explicit_direction(self) -> None:
        fundamentals, news = self._component_payloads(-75)
        result = build_local_evidence(
            technical_score=-80,
            technical_confidence=90,
            quant_validation={
                "available": True,
                "latest_score": -0.8,
                "robustness_score": 90,
                "folds": [{}, {}, {}],
                "verdict": "偏空",
            },
            fundamentals=fundamentals,
            news=news,
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLess(result["score"], 50)
        self.assertLess(result["directional_score"], 0)
        self.assertEqual(result["signal_strength"], abs(result["directional_score"]))
        self.assertIn("偏空", result["summary"])
        self.assertTrue(
            all(0 <= item["score"] <= 100 for item in result["components"])
        )

    def test_conflicting_evidence_reduces_agreement_and_confidence(self) -> None:
        aligned_fundamentals, aligned_news = self._component_payloads(80)
        aligned = build_local_evidence(
            technical_score=80,
            technical_confidence=90,
            quant_validation={
                "available": True,
                "latest_score": 0.8,
                "robustness_score": 90,
                "folds": [{}, {}, {}],
                "verdict": "偏多",
            },
            fundamentals=aligned_fundamentals,
            news=aligned_news,
        )
        conflicting_fundamentals, conflicting_news = self._component_payloads(-80)
        conflicting = build_local_evidence(
            technical_score=80,
            technical_confidence=90,
            quant_validation={
                "available": True,
                "latest_score": -0.8,
                "robustness_score": 90,
                "folds": [{}, {}, {}],
                "verdict": "分歧",
            },
            fundamentals=conflicting_fundamentals,
            news=conflicting_news,
        )
        self.assertLess(conflicting["agreement"], aligned["agreement"])
        self.assertLess(conflicting["confidence"], aligned["confidence"])
        self.assertTrue(conflicting["conflicts"])

    def test_weak_single_source_is_calibrated_toward_neutral(self) -> None:
        result = build_local_evidence(
            technical_score=80,
            technical_confidence=10,
            quant_validation={"available": False},
            fundamentals={"available": False},
            news={"available": False},
        )
        self.assertEqual(result["raw_directional_score"], 80)
        self.assertLess(result["directional_score"], result["raw_directional_score"])
        self.assertGreater(result["score"], 50)
        self.assertLess(result["score"], 90)
        self.assertLess(result["confidence"], 10)

    def test_quant_request_converts_bps_and_position_limit(self) -> None:
        config = QuantValidationRequest(
            commission_bps=4.5,
            slippage_bps=1.5,
            max_position_percent=65,
            train_rows=126,
            test_rows=42,
        ).walk_forward_config(300)
        self.assertAlmostEqual(config.commission_rate, 0.00045)
        self.assertAlmostEqual(config.slippage_rate, 0.00015)
        self.assertAlmostEqual(config.max_position, 0.65)
        self.assertEqual(config.train_rows, 126)
        self.assertEqual(config.test_rows, 42)


if __name__ == "__main__":
    unittest.main()
