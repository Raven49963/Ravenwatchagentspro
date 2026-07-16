from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .metrics import summarize_performance


@dataclass(frozen=True)
class StrategyProfile:
    key: str
    name: str
    fast_window: int
    slow_window: int
    momentum_window: int
    breakout_window: int
    rebalance_every: int
    entry_threshold: float
    exit_threshold: float
    target_volatility: float
    max_position: float

    @property
    def minimum_history(self) -> int:
        return max(
            self.slow_window,
            self.momentum_window,
            self.breakout_window,
            90,
        )


DEFAULT_PROFILES = (
    StrategyProfile(
        key="responsive",
        name="灵敏趋势",
        fast_window=10,
        slow_window=40,
        momentum_window=20,
        breakout_window=30,
        rebalance_every=3,
        entry_threshold=0.03,
        exit_threshold=-0.18,
        target_volatility=0.22,
        max_position=1.0,
    ),
    StrategyProfile(
        key="balanced",
        name="平衡多因子",
        fast_window=20,
        slow_window=60,
        momentum_window=60,
        breakout_window=55,
        rebalance_every=5,
        entry_threshold=0.05,
        exit_threshold=-0.14,
        target_volatility=0.18,
        max_position=0.9,
    ),
    StrategyProfile(
        key="defensive",
        name="防御慢速",
        fast_window=30,
        slow_window=90,
        momentum_window=90,
        breakout_window=90,
        rebalance_every=10,
        entry_threshold=0.07,
        exit_threshold=-0.10,
        target_volatility=0.14,
        max_position=0.75,
    ),
)


@dataclass(frozen=True)
class WalkForwardConfig:
    train_rows: int = 126
    test_rows: int = 42
    minimum_test_rows: int = 21
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    stress_multiplier: float = 2.0
    bootstrap_horizon: int = 63
    bootstrap_simulations: int = 1_000
    bootstrap_block_size: int = 5
    random_seed: int = 7
    max_position: float = 1.0

    @property
    def one_way_cost(self) -> float:
        return self.commission_rate + self.slippage_rate

    def validate(
        self,
        row_count: int,
        profiles: tuple[StrategyProfile, ...],
    ) -> None:
        if not profiles:
            raise ValueError("At least one strategy profile is required.")
        if self.train_rows < max(profile.minimum_history for profile in profiles) + 20:
            raise ValueError("Training window is shorter than the strategy warm-up.")
        if self.test_rows < 21 or self.minimum_test_rows < 10:
            raise ValueError("Walk-forward test windows are too short.")
        if row_count < self.train_rows + self.test_rows * 2:
            raise ValueError("At least two out-of-sample windows are required.")
        if self.commission_rate < 0 or self.slippage_rate < 0:
            raise ValueError("Trading costs cannot be negative.")
        if self.stress_multiplier < 1:
            raise ValueError("Cost stress multiplier must be at least one.")
        if self.bootstrap_horizon < 5 or self.bootstrap_simulations < 100:
            raise ValueError("Bootstrap settings are too small for a stable estimate.")
        if self.bootstrap_block_size < 1:
            raise ValueError("Bootstrap block size must be positive.")
        if self.bootstrap_block_size > self.bootstrap_horizon:
            raise ValueError("Bootstrap block size cannot exceed the horizon.")
        if not 0.05 <= self.max_position <= 1.0:
            raise ValueError("Maximum position must be between 5% and 100%.")


@dataclass(frozen=True)
class WalkForwardFold:
    number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_key: str
    selected_name: str
    selection_score: float
    train_sharpe: float
    test_return: float
    benchmark_return: float
    excess_return: float
    max_drawdown: float
    sharpe: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    stress_metrics: dict[str, float]
    robustness_score: int
    verdict: str
    summary: str
    probabilistic_sharpe: float
    fold_win_rate: float
    profile_distribution: dict[str, int]
    bootstrap: dict[str, float | int]
    equity_curve: list[dict[str, float | str]]
    latest_profile: str
    latest_score: float
    latest_position: float
    target_positions: pd.Series = field(repr=False)
    portfolio_returns: pd.Series = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": True,
            "method": "rolling_walk_forward",
            "config": {
                **asdict(self.config),
                "one_way_cost": self.config.one_way_cost,
                "execution_delay_days": 1,
                "candidate_count": len(DEFAULT_PROFILES),
            },
            "folds": [fold.to_dict() for fold in self.folds],
            "metrics": self.metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "stress_metrics": self.stress_metrics,
            "robustness_score": self.robustness_score,
            "verdict": self.verdict,
            "summary": self.summary,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "fold_win_rate": self.fold_win_rate,
            "profile_distribution": self.profile_distribution,
            "bootstrap": self.bootstrap,
            "equity_curve": self.equity_curve,
            "latest_profile": self.latest_profile,
            "latest_score": self.latest_score,
            "latest_position": self.latest_position,
            "references": [
                {
                    "title": "The Deflated Sharpe Ratio",
                    "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
                },
                {
                    "title": "The Stationary Bootstrap",
                    "url": "https://doi.org/10.1080/01621459.1994.10476870",
                },
            ],
        }


@dataclass(frozen=True)
class _ProfileSignals:
    score: pd.Series
    position: pd.Series


def adaptive_walk_forward_config(row_count: int) -> WalkForwardConfig:
    if row_count >= 420:
        return WalkForwardConfig(train_rows=252, test_rows=63)
    return WalkForwardConfig(train_rows=126, test_rows=42)


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}.")
    frame = bars.loc[:, required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if not frame.index.is_unique:
        frame = frame.loc[~frame.index.duplicated(keep="last")]
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    invalid = (frame["High"] < frame[["Open", "Close"]].max(axis=1)) | (
        frame["Low"] > frame[["Open", "Close"]].min(axis=1)
    )
    if invalid.any():
        raise ValueError("OHLC data contains inconsistent rows.")
    return frame


def _profile_signals(
    frame: pd.DataFrame,
    profile: StrategyProfile,
) -> _ProfileSignals:
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"].clip(lower=0.0)
    returns = close.pct_change()
    annual_volatility = (
        returns.rolling(20, min_periods=10).std(ddof=0) * math.sqrt(252)
    ).replace(0.0, np.nan)

    fast_average = close.rolling(profile.fast_window).mean()
    slow_average = close.rolling(profile.slow_window).mean()
    trend_scale = annual_volatility * math.sqrt(
        max(profile.slow_window - profile.fast_window, 1) / 252
    )
    trend_raw = (fast_average / slow_average - 1.0).div(trend_scale)
    trend = pd.Series(
        np.tanh(trend_raw.clip(-4.0, 4.0)), index=frame.index
    )

    momentum_return = close.pct_change(profile.momentum_window)
    momentum_scale = annual_volatility * math.sqrt(
        profile.momentum_window / 252
    )
    momentum = pd.Series(
        np.tanh(momentum_return.div(momentum_scale).clip(-4.0, 4.0)),
        index=frame.index,
    )

    previous_high = high.shift(1).rolling(profile.breakout_window).max()
    previous_low = low.shift(1).rolling(profile.breakout_window).min()
    breakout_range = (previous_high - previous_low).replace(0.0, np.nan)
    breakout = ((close - previous_low).div(breakout_range) * 2.0 - 1.0).clip(
        -1.0, 1.0
    )

    deviation = close.rolling(profile.fast_window).std(ddof=0).replace(0.0, np.nan)
    z_score = (close - fast_average).div(deviation)
    reversal = pd.Series(
        -np.tanh(z_score.clip(-5.0, 5.0) / 2.0), index=frame.index
    )

    normal_volume = volume.rolling(20, min_periods=10).median().replace(0.0, np.nan)
    volume_surprise = volume.div(normal_volume) - 1.0
    volume_confirmation = pd.Series(
        np.sign(momentum.fillna(0.0))
        * np.tanh(volume_surprise.clip(-3.0, 3.0)),
        index=frame.index,
    )

    slow_slope = slow_average.pct_change(20)
    bull = (close > slow_average) & (slow_slope > 0)
    bear = (close < slow_average) & (slow_slope < 0)
    volatility_threshold = annual_volatility.rolling(
        126, min_periods=63
    ).quantile(0.75)
    high_volatility = annual_volatility > volatility_threshold

    weights = pd.DataFrame(
        {
            "trend": 0.28,
            "momentum": 0.25,
            "breakout": 0.18,
            "reversal": 0.18,
            "volume": 0.11,
        },
        index=frame.index,
    )
    weights.loc[bull, :] = (0.32, 0.29, 0.22, 0.08, 0.09)
    weights.loc[bear, :] = (0.24, 0.20, 0.12, 0.32, 0.12)
    range_bound = ~(bull | bear)
    weights.loc[range_bound, :] = (0.18, 0.18, 0.12, 0.40, 0.12)
    weights.loc[high_volatility, :] = (0.15, 0.15, 0.10, 0.45, 0.15)

    components = pd.DataFrame(
        {
            "trend": trend,
            "momentum": momentum,
            "breakout": breakout,
            "reversal": reversal,
            "volume": volume_confirmation,
        },
        index=frame.index,
    ).fillna(0.0)
    score = (components * weights).sum(axis=1).clip(-1.0, 1.0)

    conviction = ((score - profile.entry_threshold) / 0.50).clip(0.0, 1.0)
    volatility_scale = (
        profile.target_volatility / annual_volatility
    ).clip(lower=0.20, upper=1.0)
    regime_scale = pd.Series(0.70, index=frame.index)
    regime_scale.loc[bull] = 1.0
    regime_scale.loc[bear] = 0.25
    regime_scale.loc[high_volatility] = np.minimum(
        regime_scale.loc[high_volatility], 0.55
    )
    desired = (
        conviction
        * volatility_scale.fillna(0.0)
        * regime_scale
        * profile.max_position
    ).clip(0.0, profile.max_position)

    scheduled = pd.Series(np.nan, index=frame.index, dtype=float)
    scheduled.iloc[:: profile.rebalance_every] = desired.iloc[
        :: profile.rebalance_every
    ]
    forced_exit = (score <= profile.exit_threshold) | (
        close < slow_average * 0.92
    )
    scheduled.loc[forced_exit] = 0.0
    position = scheduled.ffill().fillna(0.0)
    position.loc[slow_average.isna()] = 0.0
    return _ProfileSignals(score=score, position=position)


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


def _selection_score(
    metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
) -> float:
    sharpe = float(np.clip(metrics["sharpe"], -2.0, 2.0))
    calmar = float(np.clip(metrics["calmar"], -3.0, 3.0))
    excess = float(
        np.clip(
            metrics["annual_return"] - benchmark_metrics["annual_return"],
            -0.5,
            0.5,
        )
    )
    drawdown = min(abs(metrics["max_drawdown"]), 0.5)
    turnover = min(metrics.get("annualized_turnover", 0.0), 12.0)
    return float(
        0.50 * sharpe
        + 0.25 * calmar
        + 0.75 * excess
        - 0.35 * drawdown
        - 0.015 * turnover
    )


def probabilistic_sharpe_ratio(returns: pd.Series) -> float:
    usable = pd.to_numeric(returns, errors="coerce").dropna()
    if len(usable) < 20:
        return 0.5
    standard_deviation = float(usable.std(ddof=1))
    if standard_deviation <= 0:
        return 0.5
    sharpe = float(usable.mean() / standard_deviation)
    skewness = float(usable.skew())
    kurtosis = float(usable.kurt()) + 3.0
    denominator_squared = (
        1.0
        - skewness * sharpe
        + ((kurtosis - 1.0) / 4.0) * sharpe**2
    )
    if not math.isfinite(denominator_squared) or denominator_squared <= 0:
        return 0.5
    statistic = (
        sharpe
        * math.sqrt(len(usable) - 1)
        / math.sqrt(denominator_squared)
    )
    return float(NormalDist().cdf(statistic))


def stationary_bootstrap_risk(
    returns: pd.Series,
    *,
    horizon: int = 63,
    simulations: int = 1_000,
    block_size: int = 5,
    seed: int = 7,
) -> dict[str, float | int]:
    usable = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(float)
    if len(usable) < 20:
        raise ValueError("At least 20 out-of-sample returns are required.")
    rng = np.random.default_rng(seed)
    outcomes = np.empty(simulations, dtype=float)
    restart_probability = 1.0 / block_size
    for simulation in range(simulations):
        position = int(rng.integers(0, len(usable)))
        path = np.empty(horizon, dtype=float)
        for step in range(horizon):
            if step and rng.random() < restart_probability:
                position = int(rng.integers(0, len(usable)))
            path[step] = usable[position]
            position = (position + 1) % len(usable)
        outcomes[simulation] = float(np.prod(1.0 + path) - 1.0)
    return {
        "horizon_days": horizon,
        "simulations": simulations,
        "block_size": block_size,
        "loss_probability": round(float(np.mean(outcomes < 0.0)), 6),
        "p05_return": round(float(np.quantile(outcomes, 0.05)), 6),
        "median_return": round(float(np.quantile(outcomes, 0.50)), 6),
        "p95_return": round(float(np.quantile(outcomes, 0.95)), 6),
    }


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: round(float(value), 6) if math.isfinite(float(value)) else 0.0
        for key, value in metrics.items()
    }


def _test_starts(row_count: int, config: WalkForwardConfig) -> list[int]:
    starts = list(range(config.train_rows, row_count, config.test_rows))
    if len(starts) >= 2 and row_count - starts[-1] < config.minimum_test_rows:
        starts.pop()
    if len(starts) < 2:
        raise ValueError("At least two out-of-sample windows are required.")
    return starts


def walk_forward_validate(
    bars: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
    profiles: tuple[StrategyProfile, ...] = DEFAULT_PROFILES,
) -> WalkForwardResult:
    frame = _validated_bars(bars)
    resolved = config or adaptive_walk_forward_config(len(frame))
    resolved.validate(len(frame), profiles)
    starts = _test_starts(len(frame), resolved)
    prices = frame[["Close"]].rename(columns={"Close": "asset"})
    base_backtest = BacktestConfig(
        initial_cash=100_000.0,
        commission_rate=resolved.commission_rate,
        slippage_rate=resolved.slippage_rate,
        benchmark_symbol="asset",
    )
    stress_backtest = BacktestConfig(
        initial_cash=base_backtest.initial_cash,
        commission_rate=resolved.commission_rate * resolved.stress_multiplier,
        slippage_rate=resolved.slippage_rate * resolved.stress_multiplier,
        benchmark_symbol="asset",
    )

    signals = {profile.key: _profile_signals(frame, profile) for profile in profiles}
    candidate_results: dict[str, BacktestResult] = {}
    for profile in profiles:
        weights = signals[profile.key].position.clip(
            upper=resolved.max_position
        ).to_frame("asset")
        candidate_results[profile.key] = run_backtest(
            prices, weights, base_backtest
        )
    benchmark_weights = pd.DataFrame(1.0, index=frame.index, columns=["asset"])
    benchmark_result = run_backtest(prices, benchmark_weights, base_backtest)

    stitched_positions = pd.Series(0.0, index=frame.index, name="asset")
    selections: list[dict[str, Any]] = []
    for number, test_start in enumerate(starts, start=1):
        test_stop = starts[number] if number < len(starts) else len(frame)
        train_start = max(0, test_start - resolved.train_rows)
        benchmark_train = _period_metrics(
            benchmark_result,
            train_start,
            test_start,
            base_backtest.initial_cash,
        )
        ranked: list[tuple[float, StrategyProfile, dict[str, float]]] = []
        for profile in profiles:
            train_metrics = _period_metrics(
                candidate_results[profile.key],
                train_start,
                test_start,
                base_backtest.initial_cash,
            )
            ranked.append(
                (
                    _selection_score(train_metrics, benchmark_train),
                    profile,
                    train_metrics,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1].key), reverse=True)
        selection_score, selected, train_metrics = ranked[0]

        # Selection happens at the close before the test segment. run_backtest
        # applies that target one row later, on the first out-of-sample day.
        stitched_positions.iloc[test_start - 1 : test_stop] = signals[
            selected.key
        ].position.iloc[test_start - 1 : test_stop].clip(
            upper=resolved.max_position
        )
        selections.append(
            {
                "number": number,
                "train_start_row": train_start,
                "test_start_row": test_start,
                "test_stop_row": test_stop,
                "selected": selected,
                "selection_score": selection_score,
                "train_metrics": train_metrics,
            }
        )

    stitched_result = run_backtest(
        prices, stitched_positions.to_frame(), base_backtest
    )
    stress_result = run_backtest(
        prices, stitched_positions.to_frame(), stress_backtest
    )
    first_test = starts[0]

    folds: list[WalkForwardFold] = []
    for selection in selections:
        start = selection["test_start_row"]
        stop = selection["test_stop_row"]
        strategy_metrics = _period_metrics(
            stitched_result, start, stop, base_backtest.initial_cash
        )
        benchmark_metrics = _period_metrics(
            benchmark_result, start, stop, base_backtest.initial_cash
        )
        selected = selection["selected"]
        folds.append(
            WalkForwardFold(
                number=selection["number"],
                train_start=frame.index[selection["train_start_row"]].date().isoformat(),
                train_end=frame.index[start - 1].date().isoformat(),
                test_start=frame.index[start].date().isoformat(),
                test_end=frame.index[stop - 1].date().isoformat(),
                selected_key=selected.key,
                selected_name=selected.name,
                selection_score=round(float(selection["selection_score"]), 4),
                train_sharpe=round(
                    float(selection["train_metrics"]["sharpe"]), 4
                ),
                test_return=round(strategy_metrics["total_return"], 6),
                benchmark_return=round(benchmark_metrics["total_return"], 6),
                excess_return=round(
                    strategy_metrics["total_return"]
                    - benchmark_metrics["total_return"],
                    6,
                ),
                max_drawdown=round(strategy_metrics["max_drawdown"], 6),
                sharpe=round(strategy_metrics["sharpe"], 4),
                turnover=round(
                    strategy_metrics.get("annualized_turnover", 0.0), 6
                ),
            )
        )

    metrics = _period_metrics(
        stitched_result, first_test, len(frame), base_backtest.initial_cash
    )
    benchmark_metrics = _period_metrics(
        benchmark_result, first_test, len(frame), base_backtest.initial_cash
    )
    stress_metrics = _period_metrics(
        stress_result, first_test, len(frame), stress_backtest.initial_cash
    )
    oos_returns = stitched_result.portfolio_returns.iloc[first_test:]
    stress_returns = stress_result.portfolio_returns.iloc[first_test:]
    benchmark_returns = benchmark_result.portfolio_returns.iloc[first_test:]
    probability = probabilistic_sharpe_ratio(oos_returns)
    bootstrap = stationary_bootstrap_risk(
        oos_returns,
        horizon=resolved.bootstrap_horizon,
        simulations=resolved.bootstrap_simulations,
        block_size=resolved.bootstrap_block_size,
        seed=resolved.random_seed,
    )

    fold_win_rate = sum(fold.excess_return > 0 for fold in folds) / len(folds)
    positive_fold_rate = sum(fold.test_return > 0 for fold in folds) / len(folds)
    excess_total = metrics["total_return"] - benchmark_metrics["total_return"]
    excess_score = float(np.clip((excess_total + 0.10) / 0.30, 0.0, 1.0))
    drawdown_score = 1.0 - min(abs(metrics["max_drawdown"]) / 0.35, 1.0)
    cost_drag = max(0.0, metrics["total_return"] - stress_metrics["total_return"])
    cost_score = 1.0 - min(cost_drag / 0.05, 1.0)
    sample_score = min(len(oos_returns) / 252.0, 1.0)
    robustness_score = round(
        100
        * (
            0.25 * fold_win_rate
            + 0.20 * probability
            + 0.15 * excess_score
            + 0.15 * drawdown_score
            + 0.10 * cost_score
            + 0.10 * sample_score
            + 0.05 * positive_fold_rate
        )
    )
    if len(folds) < 3:
        robustness_score = min(robustness_score, 59)
    if robustness_score >= 70 and excess_total > 0 and probability >= 0.75:
        verdict = "较稳健"
    elif robustness_score >= 50:
        verdict = "可观察"
    else:
        verdict = "证据不足"

    selected_counts = Counter(fold.selected_name for fold in folds)
    latest_selection = selections[-1]["selected"]
    latest_signal = signals[latest_selection.key]
    latest_position = float(stitched_positions.iloc[-1])
    direction = "领先" if excess_total >= 0 else "落后"
    summary = (
        f"{len(folds)} 段滚动样本外中有 {fold_win_rate:.0%} 跑赢持有基准，"
        f"累计{direction} {abs(excess_total):.1%}；双倍交易成本后的收益为 "
        f"{stress_metrics['total_return']:.1%}。当前画像为{latest_selection.name}，"
        f"模型目标仓位 {latest_position:.0%}。"
    )

    strategy_curve = (1.0 + oos_returns).cumprod() * 100.0
    benchmark_curve = (1.0 + benchmark_returns).cumprod() * 100.0
    stress_curve = (1.0 + stress_returns).cumprod() * 100.0
    equity_curve = [
        {
            "date": timestamp.date().isoformat(),
            "strategy": round(float(strategy_curve.loc[timestamp]), 4),
            "benchmark": round(float(benchmark_curve.loc[timestamp]), 4),
            "stress": round(float(stress_curve.loc[timestamp]), 4),
        }
        for timestamp in strategy_curve.index
    ]

    return WalkForwardResult(
        config=resolved,
        folds=tuple(folds),
        metrics=_clean_metrics(metrics),
        benchmark_metrics=_clean_metrics(benchmark_metrics),
        stress_metrics=_clean_metrics(stress_metrics),
        robustness_score=robustness_score,
        verdict=verdict,
        summary=summary,
        probabilistic_sharpe=round(probability, 6),
        fold_win_rate=round(fold_win_rate, 6),
        profile_distribution=dict(selected_counts),
        bootstrap=bootstrap,
        equity_curve=equity_curve,
        latest_profile=latest_selection.name,
        latest_score=round(float(latest_signal.score.iloc[-1]), 6),
        latest_position=round(latest_position, 6),
        target_positions=stitched_positions,
        portfolio_returns=stitched_result.portfolio_returns,
    )
