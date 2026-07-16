from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskManagedMomentumConfig:
    """Parameters for long-only, volatility-aware cross-sectional momentum."""

    lookback: int = 126
    skip_recent: int = 21
    trend_window: int = 100
    volatility_window: int = 63
    rebalance_every: int = 21
    top_n: int = 2
    target_volatility: float = 0.18
    max_position: float = 0.65
    breadth_floor: float = 0.35

    def validate(self, asset_count: int) -> None:
        integer_fields = {
            "lookback": self.lookback,
            "skip_recent": self.skip_recent,
            "trend_window": self.trend_window,
            "volatility_window": self.volatility_window,
            "rebalance_every": self.rebalance_every,
            "top_n": self.top_n,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "skip_recent" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if self.top_n > asset_count:
            raise ValueError("top_n cannot exceed the number of assets.")
        if not 0 < self.target_volatility <= 1:
            raise ValueError("target_volatility must be between 0 and 1.")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be between 0 and 1.")
        if not 0 < self.breadth_floor <= 1:
            raise ValueError("breadth_floor must be between 0 and 1.")

    @property
    def minimum_history(self) -> int:
        return max(
            self.lookback + self.skip_recent,
            self.trend_window,
            self.volatility_window,
        )


@dataclass(frozen=True)
class TacticalGrowthConfig:
    """Fixed rules for a high-risk Nasdaq tactical allocation strategy."""

    signal_symbol: str = "QQQ"
    growth_symbol: str = "TQQQ"
    defensive_symbol: str = "BIL"
    momentum_horizons: tuple[int, ...] = (21, 63, 126, 252)
    required_positive: int = 2
    trend_window: int = 200
    volatility_window: int = 63
    fast_volatility_window: int = 21
    target_volatility: float = 0.55
    volatility_gate: float = 0.30
    max_growth_weight: float = 1.0

    def validate(self) -> None:
        symbols = (
            self.signal_symbol,
            self.growth_symbol,
            self.defensive_symbol,
        )
        if any(not symbol.strip() for symbol in symbols):
            raise ValueError("Tactical growth symbols cannot be empty.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Tactical growth symbols must be different.")
        if not self.momentum_horizons or any(
            horizon <= 0 for horizon in self.momentum_horizons
        ):
            raise ValueError("momentum_horizons must contain positive integers.")
        if not 1 <= self.required_positive <= len(self.momentum_horizons):
            raise ValueError(
                "required_positive must be between 1 and the number of horizons."
            )
        for name, value in {
            "trend_window": self.trend_window,
            "volatility_window": self.volatility_window,
            "fast_volatility_window": self.fast_volatility_window,
        }.items():
            if value <= 1:
                raise ValueError(f"{name} must be greater than 1.")
        if not 0 < self.target_volatility <= 2:
            raise ValueError("target_volatility must be between 0 and 2.")
        if not 0 < self.volatility_gate <= 2:
            raise ValueError("volatility_gate must be between 0 and 2.")
        if not 0 < self.max_growth_weight <= 1:
            raise ValueError("max_growth_weight must be between 0 and 1.")

    @property
    def required_symbols(self) -> tuple[str, str, str]:
        return (
            self.signal_symbol,
            self.growth_symbol,
            self.defensive_symbol,
        )

    @property
    def minimum_history(self) -> int:
        return max(
            max(self.momentum_horizons),
            self.trend_window,
            self.volatility_window,
            self.fast_volatility_window,
        )


def tactical_growth_diagnostics(
    prices: pd.DataFrame,
    config: TacticalGrowthConfig = TacticalGrowthConfig(),
) -> pd.DataFrame:
    """Calculate auditable trend, volatility, and allocation decisions."""

    config.validate()
    missing = [symbol for symbol in config.required_symbols if symbol not in prices]
    if missing:
        raise ValueError(
            "Tactical growth requires these columns: "
            + ", ".join(config.required_symbols)
            + f". Missing: {', '.join(missing)}."
        )
    if len(prices) <= config.minimum_history:
        raise ValueError(
            f"At least {config.minimum_history + 1} rows are required for "
            "tactical growth."
        )
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices index must be sorted and unique.")

    numeric_prices = prices.loc[:, list(config.required_symbols)].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_prices.isna().any().any():
        raise ValueError("prices contains missing or non-numeric values.")
    if (numeric_prices <= 0).any().any():
        raise ValueError("prices must be positive.")

    signal_price = numeric_prices[config.signal_symbol]
    returns = numeric_prices.pct_change(fill_method=None)
    momentum_votes = pd.Series(0, index=prices.index, dtype="int64")
    for horizon in config.momentum_horizons:
        momentum_votes = momentum_votes + (
            signal_price.pct_change(horizon, fill_method=None) > 0
        ).astype("int64")

    trend_average = signal_price.rolling(
        config.trend_window, min_periods=config.trend_window
    ).mean()
    growth_volatility = returns[config.growth_symbol].rolling(
        config.volatility_window, min_periods=config.volatility_window
    ).std(ddof=0) * np.sqrt(252)
    fast_volatility = returns[config.signal_symbol].rolling(
        config.fast_volatility_window,
        min_periods=config.fast_volatility_window,
    ).std(ddof=0) * np.sqrt(252)

    risk_on = (
        (momentum_votes >= config.required_positive)
        & (signal_price > trend_average)
        & (fast_volatility < config.volatility_gate)
    )
    growth_weight = (
        config.target_volatility / growth_volatility.replace(0.0, np.nan)
    ).clip(lower=0.0, upper=config.max_growth_weight)
    growth_weight = growth_weight.where(risk_on, 0.0).fillna(0.0)

    return pd.DataFrame(
        {
            "MomentumVotes": momentum_votes,
            "TrendAverage": trend_average,
            "GrowthVolatility": growth_volatility,
            "FastVolatility": fast_volatility,
            "RiskOn": risk_on.astype(bool),
            "GrowthWeight": growth_weight,
            "DefensiveWeight": 1.0 - growth_weight,
        },
        index=prices.index,
    )


def tactical_growth_allocation(
    prices: pd.DataFrame,
    config: TacticalGrowthConfig = TacticalGrowthConfig(),
) -> pd.DataFrame:
    """Allocate between a leveraged growth ETF and a defensive Treasury ETF.

    Each row uses only data available at that close. ``run_backtest`` delays the
    target by one trading day, so the signal cannot trade on the same close.
    """

    diagnostics = tactical_growth_diagnostics(prices, config)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[:, config.growth_symbol] = diagnostics["GrowthWeight"]
    weights.loc[:, config.defensive_symbol] = diagnostics["DefensiveWeight"]
    return weights


def moving_average_signals(
    prices: pd.DataFrame,
    ticker: str,
    fast: int = 5,
    slow: int = 20,
) -> pd.DataFrame:
    """Calculate double-SMA values, crossover signals, and the resulting position.

    ``Signal`` is 1 on a golden cross, -1 on a death cross, and 0 otherwise.
    The first ``slow - 1`` slow-average values are naturally NaN. During that
    warm-up period ``Position`` stays at zero, so incomplete indicators cannot
    trigger a trade.
    """

    if fast <= 0 or slow <= 0:
        raise ValueError("Moving-average windows must be positive.")
    if fast >= slow:
        raise ValueError("Use a fast window smaller than the slow window.")
    if ticker not in prices.columns:
        raise ValueError(f"Unknown ticker: {ticker}")
    if len(prices) < slow:
        raise ValueError(
            f"At least {slow} price rows are required for a {slow}-day average."
        )

    close = pd.to_numeric(prices[ticker], errors="coerce")
    fast_name = f"SMA{fast}"
    slow_name = f"SMA{slow}"
    fast_ma = close.rolling(window=fast, min_periods=fast).mean()
    slow_ma = close.rolling(window=slow, min_periods=slow).mean()
    valid = fast_ma.notna() & slow_ma.notna()
    position = ((fast_ma > slow_ma) & valid).astype(int)
    signal = position.diff().fillna(position).astype(int)

    return pd.DataFrame(
        {
            "Close": close,
            fast_name: fast_ma,
            slow_name: slow_ma,
            "Signal": signal,
            "Position": position,
        },
        index=prices.index,
    )


def moving_average_crossover(
    prices: pd.DataFrame,
    ticker: str,
    fast: int = 5,
    slow: int = 20,
) -> pd.DataFrame:
    """Hold one asset when its fast average is above its slow average."""

    signal_table = moving_average_signals(prices, ticker, fast, slow)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[:, ticker] = signal_table["Position"].astype(float)
    return weights


def momentum_rotation(
    prices: pd.DataFrame,
    lookback: int = 126,
    rebalance_every: int = 21,
    top_n: int = 2,
    require_positive: bool = True,
) -> pd.DataFrame:
    """Every few weeks, buy the strongest assets by trailing return."""

    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if top_n > len(prices.columns):
        raise ValueError("top_n cannot exceed the number of assets.")

    momentum = prices.pct_change(lookback)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    rebalance_rows = set(np.arange(lookback, len(prices), rebalance_every))

    for row_number, date in enumerate(prices.index):
        if row_number in rebalance_rows:
            scores = momentum.loc[date].dropna().sort_values(ascending=False)
            if require_positive:
                scores = scores[scores > 0]
            winners = scores.head(top_n).index
            current = pd.Series(0.0, index=prices.columns)
            if len(winners) > 0:
                current.loc[winners] = 1.0 / len(winners)

        weights.loc[date] = current

    return weights


def risk_managed_momentum(
    prices: pd.DataFrame,
    config: RiskManagedMomentumConfig = RiskManagedMomentumConfig(),
) -> pd.DataFrame:
    """Build long-only momentum weights with explicit cash and risk controls.

    Signals are calculated from closing prices available on each row. The
    backtester applies these target weights on the next row, so the strategy
    does not trade on information from the future.
    """

    if prices.empty or prices.columns.empty:
        raise ValueError("prices must contain at least one asset.")
    config.validate(len(prices.columns))
    if len(prices) <= config.minimum_history:
        raise ValueError(
            f"At least {config.minimum_history + 1} rows are required for "
            "risk-managed momentum."
        )

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce")
    if numeric_prices.isna().any().any():
        raise ValueError("prices contains missing or non-numeric values.")
    if (numeric_prices <= 0).any().any():
        raise ValueError("prices must be positive.")

    daily_returns = numeric_prices.pct_change()
    momentum = numeric_prices.shift(config.skip_recent).pct_change(config.lookback)
    trend = numeric_prices.rolling(
        config.trend_window, min_periods=config.trend_window
    ).mean()
    annual_volatility = daily_returns.rolling(
        config.volatility_window, min_periods=config.volatility_window
    ).std(ddof=0) * np.sqrt(252)

    weights = pd.DataFrame(
        np.nan, index=numeric_prices.index, columns=numeric_prices.columns
    )
    weights.iloc[: config.minimum_history] = 0.0

    for row_number in range(
        config.minimum_history, len(numeric_prices), config.rebalance_every
    ):
        date = numeric_prices.index[row_number]
        price_row = numeric_prices.loc[date]
        momentum_row = momentum.loc[date]
        volatility_row = annual_volatility.loc[date]
        trend_row = trend.loc[date]

        eligible = (
            momentum_row.notna()
            & volatility_row.notna()
            & (volatility_row > 0)
            & (momentum_row > 0)
            & (price_row > trend_row)
        )
        scores = (momentum_row / volatility_row).where(eligible).dropna()
        winners = scores.nlargest(config.top_n).index
        target = pd.Series(0.0, index=numeric_prices.columns)

        if len(winners) > 0:
            inverse_volatility = 1.0 / volatility_row.loc[winners]
            base_weights = inverse_volatility / inverse_volatility.sum()
            base_weights = base_weights.clip(upper=config.max_position)

            covariance = (
                daily_returns.loc[:date, winners]
                .tail(config.volatility_window)
                .cov()
                * 252
            )
            projected_variance = float(
                base_weights.to_numpy()
                @ covariance.to_numpy()
                @ base_weights.to_numpy()
            )
            projected_volatility = np.sqrt(max(projected_variance, 0.0))
            volatility_scale = (
                min(1.0, config.target_volatility / projected_volatility)
                if projected_volatility > 0
                else 0.0
            )

            positive_trend_breadth = float(eligible.mean())
            breadth_scale = min(
                1.0, positive_trend_breadth / config.breadth_floor
            )
            target.loc[winners] = (
                base_weights * volatility_scale * breadth_scale
            )

        weights.iloc[row_number] = target

    return weights.ffill().fillna(0.0).clip(lower=0.0, upper=1.0)
