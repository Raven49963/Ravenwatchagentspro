from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.walk_forward import (
    WalkForwardConfig,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_risk,
    walk_forward_validate,
)
from web_app import _walk_forward_payload


def synthetic_bars(rows: int = 336, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=rows)
    cycle = np.sin(np.arange(rows) / 21.0) * 0.0015
    returns = 0.00035 + cycle + rng.normal(0.0, 0.012, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    opening = close * (1.0 + rng.normal(0.0, 0.0025, rows))
    spread = rng.uniform(0.003, 0.018, rows)
    high = np.maximum(opening, close) * (1.0 + spread)
    low = np.minimum(opening, close) * (1.0 - spread)
    volume = rng.lognormal(mean=13.0, sigma=0.35, size=rows)
    return pd.DataFrame(
        {
            "Open": opening,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


class WalkForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WalkForwardConfig(
            train_rows=126,
            test_rows=42,
            bootstrap_simulations=300,
        )

    def test_walk_forward_stitches_strict_out_of_sample_folds(self) -> None:
        bars = synthetic_bars()
        result = walk_forward_validate(bars, config=self.config)

        self.assertGreaterEqual(len(result.folds), 4)
        self.assertEqual(result.folds[0].train_end, bars.index[125].date().isoformat())
        self.assertEqual(result.folds[0].test_start, bars.index[126].date().isoformat())
        self.assertTrue((result.target_positions.iloc[:125] == 0.0).all())
        self.assertTrue(result.target_positions.between(0.0, 1.0).all())
        self.assertLessEqual(
            result.stress_metrics["total_return"],
            result.metrics["total_return"] + 1e-12,
        )
        self.assertEqual(len(result.equity_curve), len(bars) - 126)

    def test_changing_future_prices_does_not_change_past_positions(self) -> None:
        bars = synthetic_bars()
        original = walk_forward_validate(bars, config=self.config)
        changed = bars.copy()
        cutoff = 250
        shock = np.linspace(1.3, 0.55, len(changed) - cutoff)
        for column in ("Open", "High", "Low", "Close"):
            changed.iloc[cutoff:, changed.columns.get_loc(column)] *= shock
        revised = walk_forward_validate(changed, config=self.config)

        pd.testing.assert_series_equal(
            original.target_positions.iloc[:cutoff],
            revised.target_positions.iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            original.portfolio_returns.iloc[:cutoff],
            revised.portfolio_returns.iloc[:cutoff],
        )

    def test_bootstrap_and_probabilistic_sharpe_are_reproducible(self) -> None:
        returns = synthetic_bars()["Close"].pct_change().dropna()
        first = stationary_bootstrap_risk(
            returns, simulations=300, horizon=42, block_size=5, seed=11
        )
        second = stationary_bootstrap_risk(
            returns, simulations=300, horizon=42, block_size=5, seed=11
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first["p05_return"], first["median_return"])
        self.assertLessEqual(first["median_return"], first["p95_return"])
        self.assertGreaterEqual(first["loss_probability"], 0.0)
        self.assertLessEqual(first["loss_probability"], 1.0)
        self.assertGreaterEqual(probabilistic_sharpe_ratio(returns), 0.0)
        self.assertLessEqual(probabilistic_sharpe_ratio(returns), 1.0)

    def test_payload_is_json_safe_and_short_history_isolated(self) -> None:
        payload = walk_forward_validate(
            synthetic_bars(), config=self.config
        ).to_dict()
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertIn("rolling_walk_forward", encoded)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["config"]["execution_delay_days"], 1)

        unavailable = _walk_forward_payload(synthetic_bars(180))
        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["status"], "insufficient_history")

    def test_configured_position_cap_is_applied_to_selection_and_oos(self) -> None:
        config = WalkForwardConfig(
            train_rows=126,
            test_rows=42,
            bootstrap_simulations=300,
            max_position=0.35,
        )
        result = walk_forward_validate(synthetic_bars(), config=config)
        self.assertLessEqual(float(result.target_positions.max()), 0.35 + 1e-12)
        self.assertLessEqual(result.latest_position, 0.35 + 1e-12)
        self.assertEqual(result.to_dict()["config"]["max_position"], 0.35)

    def test_bootstrap_block_cannot_exceed_horizon(self) -> None:
        config = WalkForwardConfig(
            train_rows=126,
            test_rows=42,
            bootstrap_horizon=21,
            bootstrap_block_size=22,
            bootstrap_simulations=300,
        )
        with self.assertRaisesRegex(ValueError, "block size"):
            walk_forward_validate(synthetic_bars(), config=config)


if __name__ == "__main__":
    unittest.main()
