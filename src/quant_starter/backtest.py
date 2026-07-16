from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    execution_delay_days: int = 1
    max_gross_exposure: float = 1.0
    benchmark_symbol: str | None = None

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.commission_rate < 0 or self.slippage_rate < 0:
            raise ValueError("commission and slippage rates cannot be negative.")
        if self.execution_delay_days < 1:
            raise ValueError("execution_delay_days must be at least 1.")
        if not 0 < self.max_gross_exposure <= 1:
            raise ValueError("max_gross_exposure must be between 0 and 1.")
        if self.benchmark_symbol is not None and not self.benchmark_symbol.strip():
            raise ValueError("benchmark_symbol cannot be blank.")


@dataclass(frozen=True)
class BacktestResult:
    prices: pd.DataFrame
    target_weights: pd.DataFrame
    asset_returns: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series
    portfolio_returns: pd.Series
    turnover: pd.Series
    equity: pd.Series
    benchmark_equity: pd.Series
    benchmark_name: str


def _validate_inputs(prices: pd.DataFrame, target_weights: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("prices is empty.")
    if target_weights.empty:
        raise ValueError("target_weights is empty.")
    if list(prices.columns) != list(target_weights.columns):
        raise ValueError("prices and target_weights must have the same columns.")
    if not prices.index.equals(target_weights.index):
        raise ValueError("prices and target_weights must have the same index.")
    if prices.isna().any().any():
        raise ValueError("prices contains missing values.")


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run a close-to-close backtest with next-day signal execution.

    target_weights is the desired allocation after looking at today's close.
    The portfolio return uses target_weights.shift(1), so today's signal affects
    tomorrow's return. This keeps the example from using future information.
    """

    _validate_inputs(prices, target_weights)
    config.validate()

    target = target_weights.copy().astype(float).clip(lower=0.0)
    row_sums = target.sum(axis=1)
    too_large = row_sums > config.max_gross_exposure
    if too_large.any():
        target.loc[too_large] = (
            target.loc[too_large]
            .div(row_sums.loc[too_large], axis=0)
            .mul(config.max_gross_exposure)
        )

    asset_returns = prices.pct_change().fillna(0.0)
    held_weights = target.shift(config.execution_delay_days).fillna(0.0)
    gross_returns = (held_weights * asset_returns).sum(axis=1)

    previous_weights = held_weights.shift(1).fillna(0.0)
    turnover = (held_weights - previous_weights).abs().sum(axis=1)
    costs = turnover * (config.commission_rate + config.slippage_rate)
    portfolio_returns = gross_returns - costs

    equity = (1.0 + portfolio_returns).cumprod() * config.initial_cash

    if config.benchmark_symbol is not None:
        if config.benchmark_symbol not in prices.columns:
            raise ValueError(
                f"Unknown benchmark symbol: {config.benchmark_symbol}."
            )
        benchmark_weights = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )
        benchmark_weights.loc[:, config.benchmark_symbol] = 1.0
        benchmark_name = config.benchmark_symbol
    else:
        benchmark_weights = pd.DataFrame(
            1.0 / len(prices.columns), index=prices.index, columns=prices.columns
        )
        benchmark_name = "Equal-weight benchmark"
    benchmark_returns = (
        benchmark_weights.shift(1).fillna(0.0) * asset_returns
    ).sum(axis=1)
    benchmark_equity = (1.0 + benchmark_returns).cumprod() * config.initial_cash

    return BacktestResult(
        prices=prices,
        target_weights=target,
        asset_returns=asset_returns,
        gross_returns=gross_returns,
        costs=costs,
        portfolio_returns=portfolio_returns,
        turnover=turnover,
        equity=equity,
        benchmark_equity=benchmark_equity,
        benchmark_name=benchmark_name,
    )
