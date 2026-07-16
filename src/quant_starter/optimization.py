from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from quant_starter.backtest import BacktestConfig, BacktestResult, run_backtest
from quant_starter.metrics import summarize_performance
from quant_starter.strategies import (
    RiskManagedMomentumConfig,
    risk_managed_momentum,
)


@dataclass(frozen=True)
class HoldoutConfig:
    training_fraction: float = 0.70
    minimum_test_rows: int = 252
    target_annual_return: float = 0.20
    max_drawdown_limit: float = 0.30

    def validate(self, row_count: int) -> None:
        if not 0.5 <= self.training_fraction <= 0.9:
            raise ValueError("training_fraction must be between 0.5 and 0.9.")
        if self.minimum_test_rows < 63:
            raise ValueError("minimum_test_rows must be at least 63.")
        if row_count < self.minimum_test_rows + 260:
            raise ValueError(
                "At least two years of daily data are required for holdout validation."
            )
        if self.target_annual_return <= 0:
            raise ValueError("target_annual_return must be positive.")
        if not 0 < self.max_drawdown_limit < 1:
            raise ValueError("max_drawdown_limit must be between 0 and 1.")


@dataclass
class HoldoutValidationResult:
    selected: RiskManagedMomentumConfig
    split_date: str
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    target_annual_return: float
    max_drawdown_limit: float
    leaderboard: pd.DataFrame
    target_weights: pd.DataFrame = field(repr=False)

    @property
    def train_target_met(self) -> bool:
        return self.train_metrics["annual_return"] >= self.target_annual_return

    @property
    def test_target_met(self) -> bool:
        return self.test_metrics["annual_return"] >= self.target_annual_return

    @property
    def test_risk_limit_met(self) -> bool:
        return abs(self.test_metrics["max_drawdown"]) <= self.max_drawdown_limit

    def summary_dict(self) -> dict[str, object]:
        return {
            "selected_parameters": asdict(self.selected),
            "split_date": self.split_date,
            "target_annual_return": self.target_annual_return,
            "max_drawdown_limit": self.max_drawdown_limit,
            "train_target_met": self.train_target_met,
            "test_target_met": self.test_target_met,
            "test_risk_limit_met": self.test_risk_limit_met,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
        }


def default_parameter_grid(asset_count: int) -> tuple[RiskManagedMomentumConfig, ...]:
    if asset_count <= 0:
        raise ValueError("asset_count must be positive.")
    top_counts = (1,) if asset_count == 1 else (1, min(2, asset_count))
    candidates = []
    for lookback in (63, 126, 189, 252):
        for trend_window in (75, 100, 150):
            for rebalance_every in (21, 42):
                for top_n in top_counts:
                    candidates.append(
                        RiskManagedMomentumConfig(
                            lookback=lookback,
                            skip_recent=21,
                            trend_window=trend_window,
                            volatility_window=63,
                            rebalance_every=rebalance_every,
                            top_n=top_n,
                            target_volatility=0.18,
                            max_position=0.65,
                            breadth_floor=0.35,
                        )
                    )
    return tuple(candidates)


def _period_metrics(
    result: BacktestResult,
    start: int,
    stop: int,
    initial_cash: float,
) -> dict[str, float]:
    returns = result.portfolio_returns.iloc[start:stop]
    turnover = result.turnover.iloc[start:stop]
    weights = result.target_weights.iloc[start:stop]
    equity = (1.0 + returns).cumprod() * initial_cash
    return summarize_performance(
        equity=equity,
        returns=returns,
        turnover=turnover,
        target_weights=weights,
    )


def _training_score(
    metrics: dict[str, float],
    config: HoldoutConfig,
) -> float:
    annual_return = metrics["annual_return"]
    target_progress = min(annual_return / config.target_annual_return, 1.0)
    drawdown_breach = max(
        0.0, abs(metrics["max_drawdown"]) - config.max_drawdown_limit
    )
    turnover_penalty = metrics.get("annualized_turnover", 0.0) * 0.01
    return float(
        metrics["sharpe"]
        + 0.30 * metrics["calmar"]
        + 0.35 * target_progress
        - 1.50 * drawdown_breach
        - turnover_penalty
    )


def optimize_risk_managed_momentum(
    prices: pd.DataFrame,
    *,
    holdout: HoldoutConfig = HoldoutConfig(),
    backtest: BacktestConfig = BacktestConfig(),
    candidates: tuple[RiskManagedMomentumConfig, ...] | None = None,
) -> HoldoutValidationResult:
    """Select parameters on a training set and report untouched holdout results."""

    holdout.validate(len(prices))
    candidate_grid = candidates or default_parameter_grid(len(prices.columns))
    if not candidate_grid:
        raise ValueError("At least one strategy candidate is required.")

    split_row = int(len(prices) * holdout.training_fraction)
    split_row = min(split_row, len(prices) - holdout.minimum_test_rows)
    maximum_warmup = max(candidate.minimum_history for candidate in candidate_grid)
    split_row = max(split_row, maximum_warmup + 126)
    if len(prices) - split_row < holdout.minimum_test_rows:
        raise ValueError("Not enough holdout rows after strategy warm-up.")

    rankings: list[dict[str, float | int]] = []
    evaluated: list[
        tuple[
            float,
            RiskManagedMomentumConfig,
            dict[str, float],
            BacktestResult,
        ]
    ] = []
    for candidate in candidate_grid:
        weights = risk_managed_momentum(prices, candidate)
        result = run_backtest(prices, weights, backtest)
        train_metrics = _period_metrics(
            result, 0, split_row, backtest.initial_cash
        )
        score = _training_score(train_metrics, holdout)
        rankings.append(
            {
                **asdict(candidate),
                "score": score,
                "train_annual_return": train_metrics["annual_return"],
                "train_sharpe": train_metrics["sharpe"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_annualized_turnover": train_metrics.get(
                    "annualized_turnover", 0.0
                ),
            }
        )
        evaluated.append((score, candidate, train_metrics, result))

    evaluated.sort(key=lambda item: item[0], reverse=True)
    _, selected, train_metrics, selected_result = evaluated[0]
    test_metrics = _period_metrics(
        selected_result,
        split_row,
        len(prices),
        backtest.initial_cash,
    )
    leaderboard = (
        pd.DataFrame(rankings)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return HoldoutValidationResult(
        selected=selected,
        split_date=prices.index[split_row].date().isoformat(),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        target_annual_return=holdout.target_annual_return,
        max_drawdown_limit=holdout.max_drawdown_limit,
        leaderboard=leaderboard,
        target_weights=selected_result.target_weights,
    )
