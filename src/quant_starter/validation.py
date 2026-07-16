from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .metrics import summarize_performance
from .strategies import (
    TacticalGrowthConfig,
    tactical_growth_allocation,
    tactical_growth_diagnostics,
)


@dataclass(frozen=True)
class TacticalPeriodResult:
    label: str
    start_date: str
    end_date: str
    rows: int
    metrics: dict[str, float]
    target_met: bool

    def summary_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rows": self.rows,
            "target_met": self.target_met,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class TacticalGrowthValidationResult:
    config: TacticalGrowthConfig
    target_annual_return: float
    split_date: str
    full_period: TacticalPeriodResult
    development_period: TacticalPeriodResult
    holdout_period: TacticalPeriodResult
    chronological_blocks: tuple[TacticalPeriodResult, ...]
    historical_target_met: bool
    stress_one_way_cost: float
    stress_full_period: TacticalPeriodResult
    stress_holdout_period: TacticalPeriodResult
    stress_blocks: tuple[TacticalPeriodResult, ...]
    cost_stress_target_met: bool
    target_weights: pd.DataFrame
    diagnostics: pd.DataFrame

    def summary_dict(self) -> dict[str, object]:
        return {
            "method": "fixed-rule chronological validation",
            "config": asdict(self.config),
            "target_annual_return": self.target_annual_return,
            "split_date": self.split_date,
            "historical_target_met": self.historical_target_met,
            "cost_stress": {
                "one_way_cost": self.stress_one_way_cost,
                "target_met": self.cost_stress_target_met,
                "full_period": self.stress_full_period.summary_dict(),
                "holdout_period": self.stress_holdout_period.summary_dict(),
                "chronological_blocks": [
                    period.summary_dict() for period in self.stress_blocks
                ],
            },
            "full_period": self.full_period.summary_dict(),
            "development_period": self.development_period.summary_dict(),
            "holdout_period": self.holdout_period.summary_dict(),
            "chronological_blocks": [
                period.summary_dict() for period in self.chronological_blocks
            ],
            "warning": (
                "Historical validation only. TQQQ is a daily leveraged ETF; "
                "future returns may be materially lower and losses may be severe."
            ),
        }


def _summarize_period(
    label: str,
    result: BacktestResult,
    dates: pd.Index,
    initial_cash: float,
    target_annual_return: float,
) -> TacticalPeriodResult:
    selected_returns = result.portfolio_returns.loc[dates]
    equity = (1.0 + selected_returns).cumprod() * initial_cash
    metrics = summarize_performance(
        equity=equity,
        returns=selected_returns,
        turnover=result.turnover.loc[dates],
        target_weights=result.target_weights.loc[dates],
    )
    return TacticalPeriodResult(
        label=label,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        rows=len(dates),
        metrics=metrics,
        target_met=metrics["annual_return"] >= target_annual_return,
    )


def validate_tactical_growth(
    prices: pd.DataFrame,
    config: TacticalGrowthConfig = TacticalGrowthConfig(),
    backtest: BacktestConfig = BacktestConfig(benchmark_symbol="QQQ"),
    target_annual_return: float = 0.20,
    development_fraction: float = 0.70,
    chronological_blocks: int = 4,
) -> TacticalGrowthValidationResult:
    """Validate fixed rules on a chronological holdout and contiguous blocks."""

    if not 0 < target_annual_return < 5:
        raise ValueError("target_annual_return must be between 0 and 5.")
    if not 0.5 <= development_fraction <= 0.9:
        raise ValueError("development_fraction must be between 0.5 and 0.9.")
    if chronological_blocks < 2:
        raise ValueError("chronological_blocks must be at least 2.")

    target_weights = tactical_growth_allocation(prices, config)
    diagnostics = tactical_growth_diagnostics(prices, config)
    result = run_backtest(prices, target_weights, backtest)

    validation_dates = prices.index[config.minimum_history :]
    minimum_rows = chronological_blocks * 126
    if len(validation_dates) < minimum_rows:
        raise ValueError(
            f"Tactical growth validation requires at least {minimum_rows} rows "
            f"after its {config.minimum_history}-row warm-up."
        )

    split_row = int(len(validation_dates) * development_fraction)
    development_dates = validation_dates[:split_row]
    holdout_dates = validation_dates[split_row:]
    block_dates = tuple(
        pd.Index(block)
        for block in np.array_split(validation_dates, chronological_blocks)
    )

    full_period = _summarize_period(
        "full_validation",
        result,
        validation_dates,
        backtest.initial_cash,
        target_annual_return,
    )
    development_period = _summarize_period(
        "development_70pct",
        result,
        development_dates,
        backtest.initial_cash,
        target_annual_return,
    )
    holdout_period = _summarize_period(
        "chronological_holdout_30pct",
        result,
        holdout_dates,
        backtest.initial_cash,
        target_annual_return,
    )
    periods = tuple(
        _summarize_period(
            f"chronological_block_{number}",
            result,
            dates,
            backtest.initial_cash,
            target_annual_return,
        )
        for number, dates in enumerate(block_dates, start=1)
    )
    historical_target_met = (
        full_period.target_met
        and holdout_period.target_met
        and all(period.target_met for period in periods)
    )

    stress_backtest = replace(
        backtest,
        commission_rate=backtest.commission_rate * 2,
        slippage_rate=backtest.slippage_rate * 2,
    )
    stress_result = run_backtest(prices, target_weights, stress_backtest)
    stress_full_period = _summarize_period(
        "cost_stress_full_validation",
        stress_result,
        validation_dates,
        stress_backtest.initial_cash,
        target_annual_return,
    )
    stress_holdout_period = _summarize_period(
        "cost_stress_holdout_30pct",
        stress_result,
        holdout_dates,
        stress_backtest.initial_cash,
        target_annual_return,
    )
    stress_periods = tuple(
        _summarize_period(
            f"cost_stress_block_{number}",
            stress_result,
            dates,
            stress_backtest.initial_cash,
            target_annual_return,
        )
        for number, dates in enumerate(block_dates, start=1)
    )
    cost_stress_target_met = (
        stress_full_period.target_met
        and stress_holdout_period.target_met
        and all(period.target_met for period in stress_periods)
    )

    return TacticalGrowthValidationResult(
        config=config,
        target_annual_return=target_annual_return,
        split_date=holdout_period.start_date,
        full_period=full_period,
        development_period=development_period,
        holdout_period=holdout_period,
        chronological_blocks=periods,
        historical_target_met=historical_target_met,
        stress_one_way_cost=(
            stress_backtest.commission_rate + stress_backtest.slippage_rate
        ),
        stress_full_period=stress_full_period,
        stress_holdout_period=stress_holdout_period,
        stress_blocks=stress_periods,
        cost_stress_target_met=cost_stress_target_met,
        target_weights=target_weights,
        diagnostics=diagnostics,
    )
