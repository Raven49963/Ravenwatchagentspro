from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_starter.agent_workflow import (
    RavenWatchAgentsWorkflow,
    WorkflowCancelled,
    WorkflowConfig,
)
from quant_starter.backtest import BacktestConfig, run_backtest
from quant_starter.data import (
    DemoMarketConfig,
    _fetch_nasdaq_api_ohlcv,
    _fetch_tencent_ohlcv_direct,
    clean_ohlcv_table,
    fetch_a_share_ohlcv,
    fetch_nasdaq_ohlcv,
    generate_demo_ohlcv,
    generate_demo_prices,
)
from quant_starter.runtime import ensure_gui_streams
from quant_starter.llm_client import LLMRequestError
from quant_starter.kline import build_kline_figure, save_kline_chart
from quant_starter.optimization import HoldoutConfig, optimize_risk_managed_momentum
from quant_starter.research_data import (
    build_research_context,
    calculate_technical_snapshot,
)
from quant_starter.research_store import (
    append_decision_memory,
    load_memory_context,
    safe_symbol_component,
    save_research_result,
)
from quant_starter.strategies import (
    RiskManagedMomentumConfig,
    TacticalGrowthConfig,
    moving_average_crossover,
    moving_average_signals,
    risk_managed_momentum,
    tactical_growth_allocation,
)
from quant_starter.symbols import (
    default_stock_choice,
    resolve_stock_choice,
    stock_choice_labels,
)
from quant_starter.validation import validate_tactical_growth


class DataTests(unittest.TestCase):
    def test_clean_ohlcv_normalizes_columns_and_dates(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2025-01-03", "2025-01-02"],
                "open": [11, 10],
                "high": [12, 11],
                "low": [10, 9],
                "close": [11.5, 10.5],
                "volume": [1200, 1000],
            }
        )
        cleaned = clean_ohlcv_table(raw)
        self.assertEqual(list(cleaned.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(str(cleaned.index[0].date()), "2025-01-02")

    def test_demo_ohlcv_is_reproducible_and_valid(self) -> None:
        first = generate_demo_ohlcv()
        second = generate_demo_ohlcv()
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first["High"] >= first[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((first["Low"] <= first[["Open", "Close"]].min(axis=1)).all())

    def test_a_share_uses_tencent_when_eastmoney_fails(self) -> None:
        class FakeAkshare:
            safe_stream_seen = False

            @staticmethod
            def stock_zh_a_hist(**_kwargs):
                raise ConnectionError("eastmoney unavailable")

            @classmethod
            def stock_zh_a_hist_tx(cls, **_kwargs):
                cls.safe_stream_seen = sys.stderr is not None
                return pd.DataFrame(
                    {
                        "date": ["2025-01-02", "2025-01-03"],
                        "open": [10.0, 10.5],
                        "high": [10.8, 11.0],
                        "low": [9.9, 10.4],
                        "close": [10.6, 10.9],
                        "amount": [1000, 1200],
                    }
                )

        with (
            mock.patch("quant_starter.data._import_optional", return_value=FakeAkshare()),
            mock.patch(
                "quant_starter.data._fetch_tencent_ohlcv_direct",
                side_effect=ConnectionError("direct unavailable"),
            ),
            mock.patch.object(sys, "stderr", None),
        ):
            bars = fetch_a_share_ohlcv("600519", "2025-01-01", "2025-01-10")
        self.assertEqual(bars.attrs["provider"], "akshare-tencent")
        self.assertEqual(len(bars), 2)
        self.assertTrue(FakeAkshare.safe_stream_seen)

    def test_tencent_direct_parses_json_without_akshare(self) -> None:
        payload = {
            "code": 0,
            "data": {
                "sz300750": {
                    "qfqday": [
                        ["2025-01-01", "10", "10.5", "10.8", "9.9", "900"],
                        ["2025-01-02", "10.5", "10.9", "11", "10.4", "1000"],
                        ["2025-01-03", "10.9", "11.2", "11.4", "10.8", "1200"],
                        ["2025-01-04", "11.2", "11.1", "11.3", "11", "800"],
                    ]
                }
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return ("kline_dayqfq2025=" + json.dumps(payload)).encode("utf-8")

        with mock.patch(
            "quant_starter.data.urlrequest.urlopen", return_value=FakeResponse()
        ):
            bars = _fetch_tencent_ohlcv_direct(
                "300750", "20250102", "20250103", "qfq"
            )
        self.assertEqual(bars.attrs["provider"], "tencent-direct")
        self.assertEqual(len(bars), 2)
        self.assertEqual(str(bars.index[0].date()), "2025-01-02")
        self.assertAlmostEqual(float(bars.iloc[-1]["Close"]), 11.2)

    def test_tencent_direct_accepts_day_key_when_adjusted_key_is_absent(self) -> None:
        payload = {
            "code": 0,
            "data": {
                "sh688981": {
                    "day": [
                        ["2025-01-02", "10", "10.5", "10.8", "9.9", "900"],
                        ["2025-01-03", "10.5", "10.9", "11", "10.4", "1000"],
                    ]
                }
            },
        }
        body = "kline_dayqfq2025=" + json.dumps(payload)
        with mock.patch("quant_starter.data._read_http_text", return_value=body):
            bars = _fetch_tencent_ohlcv_direct(
                "688981", "20250102", "20250103", "qfq"
            )
        self.assertEqual(bars.attrs["adjustment_key"], "day")
        self.assertTrue(bars.attrs["adjustment_fallback"])

    def test_a_share_failure_message_is_concise(self) -> None:
        class FailingAkshare:
            @staticmethod
            def stock_zh_a_hist(**_kwargs):
                raise ConnectionError("eastmoney unavailable " + "x" * 500)

            @staticmethod
            def stock_zh_a_hist_tx(**_kwargs):
                raise ConnectionError("tencent unavailable " + "y" * 500)

        with (
            mock.patch("quant_starter.data._import_optional", return_value=FailingAkshare()),
            mock.patch(
                "quant_starter.data._fetch_tencent_ohlcv_direct",
                side_effect=ConnectionError("direct unavailable " + "z" * 500),
            ),
        ):
            with self.assertRaises(RuntimeError) as captured:
                fetch_a_share_ohlcv("300750", "2025-01-01", "2025-01-10")
        message = str(captured.exception)
        self.assertIn("无法获取A 股 300750", message)
        self.assertIn("腾讯直连", message)
        self.assertLess(len(message), 900)

    def test_gui_runtime_supplies_missing_text_streams(self) -> None:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            ensure_gui_streams()
            usable = (
                sys.stdout is not None
                and sys.stderr is not None
                and sys.stderr.write("progress") == len("progress")
            )
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
        self.assertTrue(usable)

    def test_nasdaq_uses_stooq_when_yahoo_fails(self) -> None:
        class FakeYfinance:
            @staticmethod
            def download(**_kwargs):
                raise RuntimeError("rate limited")

        fallback = generate_demo_ohlcv().head(80)
        fallback.attrs["provider"] = "stooq"
        with (
            mock.patch("quant_starter.data._import_optional", return_value=FakeYfinance()),
            mock.patch(
                "quant_starter.data._fetch_nasdaq_api_ohlcv",
                side_effect=RuntimeError("nasdaq unavailable"),
            ),
            mock.patch(
                "quant_starter.data._fetch_yahoo_chart_ohlcv",
                side_effect=RuntimeError("chart unavailable"),
            ),
            mock.patch(
                "quant_starter.data.fetch_msn_ohlcv",
                side_effect=RuntimeError("msn unavailable"),
            ),
            mock.patch("quant_starter.data._fetch_stooq_ohlcv", return_value=fallback),
        ):
            bars = fetch_nasdaq_ohlcv("AAPL", "2025-01-01", "2025-06-01")
        self.assertEqual(bars.attrs["provider"], "stooq")

    def test_nasdaq_official_history_normalizes_currency_and_dates(self) -> None:
        payload = {
            "data": {
                "symbol": "NVDA",
                "tradesTable": {
                    "rows": [
                        {
                            "date": "07/14/2026",
                            "close": "$211.80",
                            "volume": "124,379,600",
                            "open": "$208.20",
                            "high": "$212.55",
                            "low": "$203.80",
                        },
                        {
                            "date": "07/13/2026",
                            "close": "$203.53",
                            "volume": "121,411,000",
                            "open": "$208.54",
                            "high": "$210.57",
                            "low": "$203.00",
                        },
                    ]
                },
            },
            "status": {"rCode": 200},
        }
        with mock.patch(
            "quant_starter.data._read_http_text",
            return_value=json.dumps(payload),
        ):
            bars = _fetch_nasdaq_api_ohlcv(
                "NVDA", "2026-07-01", "2026-07-15"
            )
        self.assertEqual(bars.attrs["provider"], "nasdaq-api")
        self.assertEqual(list(bars.index.strftime("%Y-%m-%d")), ["2026-07-13", "2026-07-14"])
        self.assertEqual(float(bars.iloc[-1]["Close"]), 211.8)
        self.assertEqual(float(bars.iloc[-1]["Volume"]), 124_379_600)

    def test_stock_presets_and_manual_symbols_resolve(self) -> None:
        self.assertIn("贵州茅台 · 600519", stock_choice_labels("a-share"))
        self.assertEqual(
            resolve_stock_choice("贵州茅台 · 600519", "a-share"), "600519"
        )
        self.assertEqual(resolve_stock_choice("NVDA", "nasdaq"), "NVDA")
        self.assertEqual(resolve_stock_choice("Apple", "nasdaq"), "AAPL")
        self.assertEqual(default_stock_choice("nasdaq"), "Apple · AAPL")

    def test_stock_symbol_validation_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            resolve_stock_choice("茅台", "a-share")
        with self.assertRaises(ValueError):
            resolve_stock_choice("AAPL/../../", "nasdaq")


class KlineTests(unittest.TestCase):
    def test_kline_figure_contains_candles_moving_averages_and_volume(self) -> None:
        bars = generate_demo_ohlcv("ALPHA").tail(180)
        original = bars.copy()
        figure = build_kline_figure(bars, "ALPHA", window=120)
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(len(figure.axes[0].patches), 120)
        self.assertEqual(len(figure.axes[0].lines), 3)
        self.assertEqual(len(figure.axes[1].patches), 120)
        self.assertEqual(len(figure.axes[1].get_xticklabels()), 6)
        self.assertEqual(figure.axes[1].yaxis.get_major_formatter()(1_000_000, 0), "1.0M")
        pd.testing.assert_frame_equal(bars, original)
        figure.clear()

    def test_kline_chart_is_saved_as_png(self) -> None:
        bars = generate_demo_ohlcv("ALPHA")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_kline_chart(
                bars,
                "ALPHA",
                Path(temp_dir) / "kline.png",
                window=60,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 10_000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_kline_rejects_inconsistent_ohlc(self) -> None:
        bars = generate_demo_ohlcv("ALPHA").head(80).copy()
        bars.loc[bars.index[-1], "High"] = bars.loc[bars.index[-1], "Low"]
        with self.assertRaisesRegex(ValueError, "High"):
            build_kline_figure(bars, "ALPHA")


class StrategyTests(unittest.TestCase):
    @staticmethod
    def _tactical_prices(rows: int = 900) -> pd.DataFrame:
        demo = generate_demo_prices(DemoMarketConfig(seed=23)).iloc[:rows]
        return pd.DataFrame(
            {
                "QQQ": demo["ALPHA"],
                "TQQQ": demo["GROWTH"],
                "BIL": demo["DEFENSE"],
            },
            index=demo.index,
        )

    def test_moving_average_signals_include_warmup_and_crossovers(self) -> None:
        prices = pd.DataFrame(
            {"TEST": [1, 2, 3, 4, 5, 4, 3, 2, 3, 4]},
            index=pd.date_range("2025-01-01", periods=10),
        )
        signals = moving_average_signals(prices, "TEST", fast=2, slow=3)
        self.assertTrue(pd.isna(signals.loc[prices.index[1], "SMA3"]))
        self.assertEqual(signals.loc[prices.index[2], "Signal"], 1)
        self.assertEqual(signals.loc[prices.index[6], "Signal"], -1)

    def test_backtest_uses_next_day_position(self) -> None:
        index = pd.date_range("2025-01-01", periods=3)
        prices = pd.DataFrame({"TEST": [100.0, 110.0, 121.0]}, index=index)
        weights = pd.DataFrame({"TEST": [1.0, 1.0, 1.0]}, index=index)
        result = run_backtest(
            prices,
            weights,
            BacktestConfig(initial_cash=100_000, commission_rate=0, slippage_rate=0),
        )
        self.assertAlmostEqual(result.equity.iloc[0], 100_000)
        self.assertAlmostEqual(result.equity.iloc[-1], 121_000)

    def test_crossover_weights_hold_only_selected_asset(self) -> None:
        index = pd.date_range("2025-01-01", periods=30)
        prices = pd.DataFrame(
            {"A": range(1, 31), "B": range(31, 61)}, index=index, dtype=float
        )
        weights = moving_average_crossover(prices, "A", fast=5, slow=20)
        self.assertEqual(float(weights["B"].sum()), 0.0)
        self.assertEqual(float(weights.iloc[-1]["A"]), 1.0)

    def test_risk_managed_momentum_respects_cash_and_position_limits(self) -> None:
        prices = generate_demo_prices(DemoMarketConfig(seed=11)).iloc[:700]
        config = RiskManagedMomentumConfig(
            lookback=63,
            skip_recent=5,
            trend_window=50,
            volatility_window=20,
            rebalance_every=21,
            top_n=2,
            target_volatility=0.20,
            max_position=0.55,
        )
        weights = risk_managed_momentum(prices, config)
        self.assertTrue((weights >= 0).all().all())
        self.assertTrue((weights <= 0.55 + 1e-12).all().all())
        self.assertTrue((weights.sum(axis=1) <= 1.0 + 1e-12).all())
        self.assertEqual(
            float(weights.iloc[: config.minimum_history].sum().sum()), 0.0
        )
        self.assertGreater(float(weights.sum().sum()), 0.0)

    def test_risk_managed_momentum_does_not_use_future_prices(self) -> None:
        prices = generate_demo_prices(DemoMarketConfig(seed=5)).iloc[:800]
        config = RiskManagedMomentumConfig(
            lookback=63,
            skip_recent=5,
            trend_window=50,
            volatility_window=20,
            rebalance_every=21,
            top_n=2,
        )
        changed_future = prices.copy()
        changed_future.iloc[600:] = changed_future.iloc[600:] * 4.0
        original = risk_managed_momentum(prices, config)
        changed = risk_managed_momentum(changed_future, config)
        pd.testing.assert_frame_equal(original.iloc[:600], changed.iloc[:600])

    def test_tactical_growth_weights_are_fully_allocated(self) -> None:
        prices = self._tactical_prices()
        weights = tactical_growth_allocation(prices)
        self.assertTrue((weights >= 0).all().all())
        self.assertTrue((weights <= 1).all().all())
        self.assertTrue((weights.sum(axis=1) - 1.0).abs().lt(1e-12).all())
        self.assertEqual(float(weights["QQQ"].sum()), 0.0)

    def test_tactical_growth_does_not_use_future_prices(self) -> None:
        prices = self._tactical_prices()
        changed_future = prices.copy()
        changed_future.iloc[650:] = changed_future.iloc[650:] * 3.0
        original = tactical_growth_allocation(prices)
        changed = tactical_growth_allocation(changed_future)
        pd.testing.assert_frame_equal(original.iloc[:650], changed.iloc[:650])

    def test_tactical_growth_requires_fixed_universe(self) -> None:
        prices = self._tactical_prices().drop(columns="BIL")
        with self.assertRaisesRegex(ValueError, "Missing: BIL"):
            tactical_growth_allocation(prices)

    def test_tactical_growth_chronological_validation(self) -> None:
        prices = self._tactical_prices(rows=1400)
        config = TacticalGrowthConfig(
            momentum_horizons=(5, 10, 20, 40),
            trend_window=40,
            volatility_window=20,
            fast_volatility_window=10,
            volatility_gate=1.5,
        )
        backtest = BacktestConfig(
            commission_rate=0.0003,
            slippage_rate=0.0002,
            benchmark_symbol="QQQ",
        )
        validation = validate_tactical_growth(
            prices,
            config=config,
            backtest=backtest,
        )
        self.assertEqual(len(validation.chronological_blocks), 4)
        self.assertEqual(
            validation.split_date,
            validation.holdout_period.start_date,
        )
        self.assertGreater(validation.holdout_period.rows, 0)
        self.assertIn("annual_return", validation.full_period.metrics)
        self.assertEqual(validation.stress_one_way_cost, 0.001)
        self.assertEqual(len(validation.stress_blocks), 4)
        result = run_backtest(prices, validation.target_weights, backtest)
        self.assertEqual(result.benchmark_name, "QQQ")

    def test_holdout_optimizer_reports_untouched_test_period(self) -> None:
        prices = generate_demo_prices(DemoMarketConfig(seed=7))
        candidates = (
            RiskManagedMomentumConfig(
                lookback=63,
                trend_window=75,
                rebalance_every=21,
                top_n=1,
            ),
            RiskManagedMomentumConfig(
                lookback=126,
                trend_window=100,
                rebalance_every=42,
                top_n=2,
            ),
        )
        validation = optimize_risk_managed_momentum(
            prices,
            holdout=HoldoutConfig(minimum_test_rows=252),
            candidates=candidates,
        )
        self.assertIn(validation.selected, candidates)
        self.assertEqual(len(validation.target_weights), len(prices))
        self.assertEqual(len(validation.leaderboard), 2)
        self.assertIn("annual_return", validation.test_metrics)
        self.assertIn("sortino", validation.test_metrics)


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = build_research_context(
            source="demo",
            symbol="ALPHA",
            start="2024-01-01",
            end="2025-12-31",
        )

    def test_technical_snapshot_has_auditable_fields(self) -> None:
        snapshot = calculate_technical_snapshot(self.context.bars)
        for key in ("close", "sma5", "sma20", "rsi14", "macd", "max_drawdown"):
            self.assertIn(key, snapshot)
            self.assertIsNotNone(snapshot[key])

    def test_offline_workflow_runs_full_team(self) -> None:
        events = []
        workflow = RavenWatchAgentsWorkflow(
            WorkflowConfig(mode="offline", debate_rounds=2, risk_rounds=2)
        )
        result = workflow.run(self.context, progress=events.append)
        self.assertIn(result.decision.action, {"BUY", "HOLD", "SELL"})
        self.assertIn("portfolio_manager", result.reports)
        self.assertIn("bull_2", result.reports)
        self.assertIn("risk_conservative_2", result.reports)
        self.assertEqual(events[-1].step, events[-1].total)
        self.assertEqual(len(result.agent_runs), events[-1].total)
        self.assertTrue(all(item.status == "completed" for item in result.agent_runs))

    def test_workflow_can_be_cancelled_before_first_agent(self) -> None:
        cancel = threading.Event()
        cancel.set()
        workflow = RavenWatchAgentsWorkflow(WorkflowConfig(mode="offline"))
        with self.assertRaises(WorkflowCancelled):
            workflow.run(self.context, cancel_event=cancel)

    def test_online_workflow_transparently_falls_back(self) -> None:
        class FailingClient:
            @staticmethod
            def complete(_system: str, _user: str) -> str:
                raise LLMRequestError("test endpoint unavailable")

        workflow = RavenWatchAgentsWorkflow(
            WorkflowConfig(
                mode="online",
                selected_analysts=("market",),
                fallback_to_offline=True,
            ),
            FailingClient(),  # type: ignore[arg-type]
        )
        result = workflow.run(self.context)
        self.assertIn(result.decision.action, {"BUY", "HOLD", "SELL"})
        self.assertGreaterEqual(len(result.warnings), 1)
        self.assertIn("离线", result.reports["market"])
        self.assertTrue(all(item.status == "fallback" for item in result.agent_runs))

    def test_online_workflow_isolates_one_agent_failure(self) -> None:
        class PartiallyFailingClient:
            @staticmethod
            def complete(system: str, _user: str) -> str:
                if "情绪分析师" in system:
                    raise RuntimeError("sentiment provider timeout")
                if "组合经理" in system:
                    return json.dumps(
                        {
                            "action": "HOLD",
                            "confidence": 61,
                            "target_allocation": 0.35,
                            "stop_loss_pct": 0.07,
                            "take_profit_pct": 0.14,
                            "time_horizon": "1-3个月",
                            "rationale": "测试在线组合决策",
                        },
                        ensure_ascii=False,
                    )
                return "# 在线报告\n\n测试角色已完成。"

        events = []
        workflow = RavenWatchAgentsWorkflow(
            WorkflowConfig(mode="online", fallback_to_offline=True),
            PartiallyFailingClient(),  # type: ignore[arg-type]
        )
        result = workflow.run(self.context, progress=events.append)
        self.assertEqual(len(result.agent_runs), 12)
        sentiment = next(
            item for item in result.agent_runs if item.agent_id == "sentiment"
        )
        self.assertEqual(sentiment.status, "fallback")
        self.assertEqual(sentiment.execution_mode, "fallback")
        self.assertEqual(result.decision.confidence, 61)
        self.assertTrue(
            all(
                item.status == "completed"
                for item in result.agent_runs
                if item.agent_id != "sentiment"
            )
        )
        self.assertTrue(any(event.status == "fallback" for event in events))

    def test_optional_a_share_details_failure_stays_concise(self) -> None:
        bars = generate_demo_ohlcv("ALPHA").tail(180)
        bars.attrs["provider"] = "tencent-direct"
        with (
            mock.patch(
                "quant_starter.research_data.fetch_a_share_ohlcv",
                return_value=bars,
            ),
            mock.patch(
                "quant_starter.research_data._run_with_timeout",
                side_effect=ConnectionError("blocked " + "x" * 1000),
            ),
        ):
            context = build_research_context(
                source="a-share",
                symbol="300750",
                start="2025-01-01",
                end="2025-12-31",
                fetch_details=True,
            )
        detail_warning = next(
            warning for warning in context.warnings if "基本面/新闻" in warning
        )
        self.assertLess(len(detail_warning), 280)
        self.assertEqual(
            context.technical["data_provider_label"], "腾讯证券直连"
        )

    def test_report_bundle_and_decision_memory(self) -> None:
        workflow = RavenWatchAgentsWorkflow(WorkflowConfig(mode="offline"))
        result = workflow.run(self.context)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = save_research_result(result, root / "reports")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["symbol"], "ALPHA")
            self.assertTrue((run_dir / "full_report.md").exists())
            self.assertTrue((run_dir / "research_overview.png").exists())
            self.assertTrue((run_dir / "kline_chart.png").exists())
            self.assertGreaterEqual(len(list((run_dir / "reports").glob("*.md"))), 10)

            memory_file = root / "memory" / "decision_log.jsonl"
            append_decision_memory(result, memory_file)
            memory = load_memory_context(
                "ALPHA", memory_file, current_close=float(result.context.technical["close"]) * 1.1
            )
            self.assertIn(result.decision.action, memory)
            self.assertIn("+10.00%", memory)

    def test_safe_symbol_rejects_path_traversal(self) -> None:
        self.assertEqual(safe_symbol_component("../AAPL"), "AAPL")
        with self.assertRaises(ValueError):
            safe_symbol_component("../..")


if __name__ == "__main__":
    unittest.main()
