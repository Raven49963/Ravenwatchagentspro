from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSignal:
    key: str
    name: str
    category: str
    value: float
    score: float
    weight: float
    direction: str
    description: str
    formula: str = ""
    reference_title: str = ""
    reference_url: str = ""
    history_required: int = 1
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketRegime:
    key: str
    name: str
    trend: str
    volatility: str
    confidence: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategySignal:
    key: str
    name: str
    score: float
    weight: float
    action: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorMiningResult:
    key: str
    name: str
    information_coefficient: float
    t_statistic: float
    directional_win_rate: float
    observations: int
    horizon_days: int
    category: str = ""
    reference_title: str = ""
    reference_url: str = ""
    fold_information_coefficients: tuple[float, ...] = ()
    stability_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompositeResearch:
    score: float
    directional_score: float
    raw_directional_score: float
    signal_strength: float
    agreement: float
    factor_stability: float
    factor_coverage: float
    calibration_factor: float
    action: str
    action_label: str
    confidence: int
    target_position: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_level: str
    regime: MarketRegime
    factors: tuple[FactorSignal, ...]
    strategies: tuple[StrategySignal, ...]
    factor_mining: tuple[FactorMiningResult, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["regime"] = self.regime.to_dict()
        result["factors"] = [item.to_dict() for item in self.factors]
        result["strategies"] = [item.to_dict() for item in self.strategies]
        result["factor_mining"] = [item.to_dict() for item in self.factor_mining]
        return result


FACTOR_WEIGHTS = {
    "trend_alignment": 0.08,
    "trend_slope": 0.05,
    "momentum_5d": 0.04,
    "momentum_20d": 0.06,
    "momentum_60d": 0.05,
    "breakout_20d": 0.05,
    "rsi_reversal": 0.05,
    "bollinger_reversal": 0.05,
    "volatility_quality": 0.05,
    "drawdown_quality": 0.05,
    "volume_confirmation": 0.04,
    "obv_trend": 0.04,
    "relative_strength": 0.03,
    "momentum_12_1": 0.05,
    "amihud_liquidity": 0.04,
    "parkinson_quality": 0.04,
    "downside_quality": 0.04,
    "money_flow_reversal": 0.03,
    "directional_movement": 0.03,
    "overnight_strength": 0.02,
    "intraday_strength": 0.02,
    "momentum_vol_adjusted": 0.015,
    "trend_efficiency": 0.015,
    "high_52w_proximity": 0.015,
    "expected_shortfall_quality": 0.015,
    "ulcer_quality": 0.015,
    "gap_risk_quality": 0.015,
}


FACTOR_REFERENCE_CATALOG: dict[str, dict[str, Any]] = {
    "momentum_12_1": {
        "name": "12-1月动量",
        "category": "动量",
        "history_required": 253,
        "formula": "P(t-21) / P(t-252) - 1",
        "reference_title": "Kenneth French Data Library - Momentum Factor",
        "reference_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_mom_factor.html",
    },
    "amihud_liquidity": {
        "name": "Amihud流动性",
        "category": "流动性",
        "history_required": 40,
        "formula": "mean(|R| / (Close * Volume), 20)",
        "reference_title": "Amihud (2002) Illiquidity and Stock Returns",
        "reference_url": "https://doi.org/10.1016/S1386-4181(01)00024-6",
    },
    "parkinson_quality": {
        "name": "Parkinson区间波动",
        "category": "风险",
        "history_required": 40,
        "formula": "sqrt(252 * mean(log(High/Low)^2) / (4*log(2)))",
        "reference_title": "Parkinson (1980) Extreme Value Variance",
        "reference_url": "https://doi.org/10.1086/296071",
    },
    "downside_quality": {
        "name": "下行风险质量",
        "category": "风险",
        "history_required": 40,
        "formula": "sqrt(252 * mean(min(R, 0)^2, 20))",
        "reference_title": "Sortino & Price (1994) Downside Risk Framework",
        "reference_url": "https://doi.org/10.3905/joi.3.3.59",
    },
    "money_flow_reversal": {
        "name": "MFI资金反转",
        "category": "资金流",
        "history_required": 30,
        "formula": "100 - 100 / (1 + positive_money_flow / negative_money_flow)",
        "reference_title": "TA-Lib Money Flow Index",
        "reference_url": "https://ta-lib.github.io/ta-doc/indicator/MFI.htm",
    },
    "directional_movement": {
        "name": "ADX方向强度",
        "category": "趋势",
        "history_required": 40,
        "formula": "(PLUS_DI - MINUS_DI) * ADX / 100",
        "reference_title": "TA-Lib Average Directional Movement Index",
        "reference_url": "https://ta-lib.github.io/ta-doc/indicator/ADX.htm",
    },
    "overnight_strength": {
        "name": "隔夜收益强度",
        "category": "时段结构",
        "history_required": 30,
        "formula": "prod(Open(t) / Close(t-1), 20) - 1",
        "reference_title": "Berkman et al. (2012) Paying Attention",
        "reference_url": "https://doi.org/10.1017/S0022109012000270",
    },
    "intraday_strength": {
        "name": "日内收益强度",
        "category": "时段结构",
        "history_required": 30,
        "formula": "prod(Close(t) / Open(t), 20) - 1",
        "reference_title": "Berkman et al. (2012) Paying Attention",
        "reference_url": "https://doi.org/10.1017/S0022109012000270",
    },
    "momentum_vol_adjusted": {
        "name": "波动调整动量",
        "category": "动量",
        "history_required": 126,
        "formula": "R(63) / annualized_volatility(63)",
        "reference_title": "Moreira & Muir (2017) Volatility-Managed Portfolios",
        "reference_url": "https://www.nber.org/papers/w22208",
    },
    "trend_efficiency": {
        "name": "趋势效率",
        "category": "趋势",
        "history_required": 40,
        "formula": "(P(t)-P(t-20)) / sum(|delta P|, 20)",
        "reference_title": "Kaufman efficiency-ratio trend filter",
        "reference_url": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama",
    },
    "high_52w_proximity": {
        "name": "52周高点接近度",
        "category": "突破",
        "history_required": 253,
        "formula": "P(t) / max(P, 252) - 1",
        "reference_title": "George & Hwang (2004) The 52-Week High and Momentum Investing",
        "reference_url": "https://doi.org/10.1111/j.1540-6261.2004.00695.x",
    },
    "expected_shortfall_quality": {
        "name": "期望损失质量",
        "category": "风险",
        "history_required": 126,
        "formula": "-mean(R | R <= quantile(R, 5%), 63) * sqrt(252)",
        "reference_title": "Basel Committee Minimum Capital Requirements for Market Risk",
        "reference_url": "https://www.bis.org/bcbs/publ/d457.htm",
    },
    "ulcer_quality": {
        "name": "持续回撤质量",
        "category": "风险",
        "history_required": 126,
        "formula": "sqrt(mean((P / rolling_max(P, 63) - 1)^2, 63))",
        "reference_title": "Martin & McCann Ulcer Index downside-risk framework",
        "reference_url": "https://doi.org/10.1007/s00291-023-00719-x",
    },
    "gap_risk_quality": {
        "name": "隔夜跳空风险",
        "category": "风险",
        "history_required": 60,
        "formula": "std(Open(t) / Close(t-1) - 1, 20) * sqrt(252)",
        "reference_title": "Berkman et al. (2012) Paying Attention",
        "reference_url": "https://doi.org/10.1017/S0022109012000270",
    },
}


def factor_reference_catalog() -> list[dict[str, Any]]:
    return [
        {"key": key, **metadata}
        for key, metadata in FACTOR_REFERENCE_CATALOG.items()
    ]


STRATEGY_NAMES = {
    "trend": "趋势跟随",
    "momentum": "多周期动量",
    "breakout": "放量突破",
    "mean_reversion": "均值回归",
    "volume_price": "量价确认",
}


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError("因子分析缺少字段：" + "、".join(sorted(missing)))
    frame = bars.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 80:
        raise ValueError("专业因子分析至少需要 80 根有效 K 线。")
    if (frame["Close"] <= 0).any():
        raise ValueError("收盘价必须为正数。")
    return frame


def _clip(value: float, limit: float = 100.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(-limit, min(limit, value)))


def _rating_score(directional_score: float) -> float:
    """Map a signed market direction to an unambiguous 0-100 rating."""
    return float(max(0.0, min(100.0, 50.0 + _clip(directional_score) / 2.0)))


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    if left_rank.nunique(dropna=True) < 2 or right_rank.nunique(dropna=True) < 2:
        return 0.0
    value = float(left_rank.corr(right_rank))
    return value if math.isfinite(value) else 0.0


def _direction_phrase(score: float) -> str:
    if score >= 25:
        return "多头方向较强"
    if score >= 8:
        return "略偏多"
    if score > -8:
        return "方向中性"
    if score > -25:
        return "略偏空"
    return "空头方向较强"


def _latest(series: pd.Series, default: float = 0.0) -> float:
    usable = pd.to_numeric(series, errors="coerce").dropna()
    return float(usable.iloc[-1]) if not usable.empty else default


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain.div(loss.replace(0, np.nan))
    result = 100 - 100 / (1 + relative_strength)
    return result.fillna(50.0)


def _money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    movement = typical.diff()
    positive = raw_flow.where(movement > 0, 0.0).rolling(window).sum()
    negative = raw_flow.where(movement < 0, 0.0).rolling(window).sum()
    ratio = positive.div(negative.replace(0, np.nan))
    result = 100 - 100 / (1 + ratio)
    result = result.where(negative != 0, 100.0)
    result = result.where(~((positive == 0) & (negative == 0)), 50.0)
    return result.fillna(50.0)


def _directional_strength(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    upward = high.diff()
    downward = -low.diff()
    plus_dm = pd.Series(
        np.where((upward > downward) & (upward > 0), upward, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((downward > upward) & (downward > 0), downward, 0.0),
        index=close.index,
    )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high.sub(low),
            high.sub(previous_close).abs(),
            low.sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1 / window
    average_range = true_range.ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean().div(average_range.replace(0, np.nan))
    minus_di = 100 * minus_dm.ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean().div(average_range.replace(0, np.nan))
    dx = 100 * plus_di.sub(minus_di).abs().div(
        plus_di.add(minus_di).replace(0, np.nan)
    )
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    return plus_di.sub(minus_di).mul(adx.div(100)).fillna(0.0)


def _rolling_percentile(series: pd.Series, value: float) -> float:
    usable = pd.to_numeric(series.tail(120), errors="coerce").dropna()
    if usable.empty or not math.isfinite(value):
        return 0.5
    return float((usable <= value).mean())


def _rolling_compound(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.add(1.0).rolling(window).apply(np.prod, raw=True).sub(1.0)


def _linear_slope(values: pd.Series, window: int = 20) -> float:
    usable = pd.to_numeric(values.tail(window), errors="coerce").dropna()
    if len(usable) < max(5, window // 2):
        return 0.0
    y = np.log(usable.to_numpy(dtype=float))
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0] * 252)


def _direction(score: float) -> str:
    if score >= 20:
        return "bullish"
    if score <= -20:
        return "bearish"
    return "neutral"


def _action(score: float) -> str:
    if score >= 30:
        return "BUY"
    if score <= -30:
        return "SELL"
    return "HOLD"


def _action_label(score: float) -> str:
    if score >= 60:
        return "强势看多"
    if score >= 25:
        return "偏多"
    if score <= -60:
        return "强势看空"
    if score <= -25:
        return "偏空"
    return "中性观察"


def _factor_signal(
    key: str,
    name: str,
    category: str,
    value: float,
    score: float,
    description: str,
    *,
    available: bool = True,
) -> FactorSignal:
    metadata = FACTOR_REFERENCE_CATALOG.get(key, {})
    normalized = _clip(score) if available else 0.0
    return FactorSignal(
        key=key,
        name=name,
        category=category,
        value=float(value),
        score=round(normalized, 2),
        weight=FACTOR_WEIGHTS[key],
        direction=_direction(normalized),
        description=description,
        formula=str(metadata.get("formula", "")),
        reference_title=str(metadata.get("reference_title", "")),
        reference_url=str(metadata.get("reference_url", "")),
        history_required=int(metadata.get("history_required", 1)),
        available=available,
    )


def calculate_factor_signals(
    bars: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> tuple[FactorSignal, ...]:
    frame = _validated_bars(bars)
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"].clip(lower=0)
    returns = close.pct_change()

    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    std20 = close.rolling(20).std(ddof=0)
    ret5 = _latest(close.pct_change(5))
    ret20 = _latest(close.pct_change(20))
    ret60 = _latest(close.pct_change(60))
    momentum_12_1_available = len(close) >= 253
    momentum_12_1 = (
        float(close.iloc[-22] / close.iloc[-253] - 1)
        if momentum_12_1_available
        else 0.0
    )
    trend_gap20 = _latest(close.div(sma20).sub(1))
    trend_gap60 = _latest(close.div(sma60).sub(1))
    trend_alignment = (trend_gap20 * 0.6 + trend_gap60 * 0.4)
    annual_slope = _linear_slope(close, 20)
    rsi14 = _latest(_rsi(close))
    money_flow_index = _latest(_money_flow_index(high, low, close, volume), 50.0)
    directional_strength = _latest(_directional_strength(high, low, close))
    bollinger_z = _latest(close.sub(sma20).div(std20.replace(0, np.nan)))
    realized_volatility = _latest(returns.rolling(20).std(ddof=0)) * math.sqrt(252)

    rolling_volatility = returns.rolling(20).std(ddof=0) * math.sqrt(252)
    recent_volatility = rolling_volatility.tail(120).dropna()
    if recent_volatility.empty:
        volatility_percentile = 0.5
    else:
        volatility_percentile = float(
            (recent_volatility <= realized_volatility).mean()
        )

    rolling_peak = close.rolling(min(252, len(close)), min_periods=20).max()
    drawdown = _latest(close.div(rolling_peak).sub(1))

    valid_high = high.where(high > 0)
    valid_low = low.where(low > 0)
    range_variance = (
        np.log(valid_high.div(valid_low))
        .pow(2)
        .div(4 * math.log(2))
    )
    parkinson_series = range_variance.rolling(20).mean().mul(252).pow(0.5)
    parkinson_volatility = _latest(parkinson_series)
    parkinson_percentile = _rolling_percentile(
        parkinson_series, parkinson_volatility
    )

    downside_series = (
        returns.clip(upper=0).pow(2).rolling(20).mean().mul(252).pow(0.5)
    )
    downside_volatility = _latest(downside_series)
    downside_percentile = _rolling_percentile(
        downside_series, downside_volatility
    )
    volatility_63 = returns.rolling(63).std(ddof=0).mul(math.sqrt(252))
    momentum_vol_series = close.pct_change(63).div(
        volatility_63.replace(0, np.nan)
    )
    momentum_vol_adjusted = _latest(momentum_vol_series)
    momentum_vol_available = len(close) >= 126 and math.isfinite(momentum_vol_adjusted)

    path_length_20 = close.diff().abs().rolling(20).sum()
    trend_efficiency_series = close.diff(20).div(path_length_20.replace(0, np.nan))
    trend_efficiency = _latest(trend_efficiency_series)

    high_52w_available = len(close) >= 253
    high_52w_series = close.div(close.rolling(252).max()).sub(1)
    high_52w_proximity = _latest(high_52w_series) if high_52w_available else 0.0

    def expected_shortfall(window: np.ndarray) -> float:
        finite = window[np.isfinite(window)]
        if len(finite) < 20:
            return np.nan
        threshold = float(np.quantile(finite, 0.05))
        tail = finite[finite <= threshold]
        return float(-tail.mean() * math.sqrt(252)) if len(tail) else np.nan

    expected_shortfall_series = returns.rolling(63).apply(
        expected_shortfall,
        raw=True,
    )
    expected_shortfall_value = _latest(expected_shortfall_series)
    expected_shortfall_available = len(close) >= 126
    expected_shortfall_percentile = _rolling_percentile(
        expected_shortfall_series, expected_shortfall_value
    )

    drawdown_63 = close.div(close.rolling(63).max()).sub(1)
    ulcer_series = drawdown_63.pow(2).rolling(63).mean().pow(0.5)
    ulcer_value = _latest(ulcer_series)
    ulcer_available = len(close) >= 126
    ulcer_percentile = _rolling_percentile(ulcer_series, ulcer_value)
    high20 = _latest(high.rolling(20).max(), _latest(high))
    low20 = _latest(low.rolling(20).min(), _latest(low))
    close_latest = _latest(close)
    breakout_position = (
        (close_latest - low20) / max(high20 - low20, close_latest * 0.001)
    )

    volume_average = _latest(volume.rolling(20).mean(), 1.0)
    volume_ratio = _latest(volume) / max(volume_average, 1.0)
    signed_volume = np.sign(close.diff().fillna(0.0)) * volume
    obv = signed_volume.cumsum()
    obv_scale = max(float(volume.tail(20).mean()) * 20, 1.0)
    obv_change = float(obv.diff(20).iloc[-1]) / obv_scale

    dollar_volume = close.mul(volume).where(volume > 0)
    amihud_daily = returns.abs().div(dollar_volume.replace(0, np.nan)).mul(1e8)
    amihud_series = amihud_daily.rolling(20).mean()
    amihud_value = _latest(amihud_series)
    amihud_available = int(dollar_volume.tail(40).notna().sum()) >= 20
    amihud_percentile = _rolling_percentile(amihud_series, amihud_value)

    overnight_returns = frame["Open"].div(close.shift(1)).sub(1)
    intraday_returns = close.div(frame["Open"].replace(0, np.nan)).sub(1)
    overnight_series = _rolling_compound(overnight_returns)
    intraday_series = _rolling_compound(intraday_returns)
    overnight_strength = _latest(overnight_series)
    intraday_strength = _latest(intraday_series)
    overnight_available = int(overnight_returns.tail(30).notna().sum()) >= 20
    intraday_available = int(intraday_returns.tail(30).notna().sum()) >= 20
    gap_risk_series = overnight_returns.rolling(20).std(ddof=0).mul(math.sqrt(252))
    gap_risk = _latest(gap_risk_series)
    gap_risk_available = int(overnight_returns.tail(60).notna().sum()) >= 40
    gap_risk_percentile = _rolling_percentile(gap_risk_series, gap_risk)

    if benchmark is not None:
        benchmark_close = pd.to_numeric(benchmark, errors="coerce").dropna()
        aligned = pd.concat(
            [close.rename("asset"), benchmark_close.rename("benchmark")], axis=1
        ).dropna()
        if len(aligned) >= 21:
            relative_strength = float(
                aligned["asset"].pct_change(20).iloc[-1]
                - aligned["benchmark"].pct_change(20).iloc[-1]
            )
        else:
            relative_strength = ret20
    else:
        relative_strength = ret20

    signals = (
        _factor_signal(
            "trend_alignment",
            "均线排列",
            "趋势",
            trend_alignment,
            trend_alignment * 1600,
            "现价相对 20 日与 60 日均线的加权偏离。",
        ),
        _factor_signal(
            "trend_slope",
            "趋势斜率",
            "趋势",
            annual_slope,
            annual_slope * 90,
            "近 20 日对数价格趋势的年化斜率。",
        ),
        _factor_signal(
            "directional_movement",
            "ADX方向强度",
            "趋势",
            directional_strength,
            directional_strength * 2.2,
            "用 +DI、-DI 与 ADX 同时衡量趋势方向和强度。",
        ),
        _factor_signal(
            "trend_efficiency",
            "趋势效率",
            "趋势",
            trend_efficiency,
            trend_efficiency * 105,
            "比较 20 日净位移与逐日路径长度，降低震荡噪声对趋势判断的干扰。",
        ),
        _factor_signal(
            "momentum_5d",
            "5日动量",
            "动量",
            ret5,
            ret5 * 900,
            "近 5 个交易日收益率。",
        ),
        _factor_signal(
            "momentum_20d",
            "20日动量",
            "动量",
            ret20,
            ret20 * 520,
            "近 20 个交易日收益率。",
        ),
        _factor_signal(
            "momentum_60d",
            "60日动量",
            "动量",
            ret60,
            ret60 * 300,
            "近 60 个交易日收益率。",
        ),
        _factor_signal(
            "momentum_vol_adjusted",
            "波动调整动量",
            "动量",
            momentum_vol_adjusted,
            momentum_vol_adjusted * 135,
            (
                "用 63 日年化波动调整同期收益，避免高波动上涨获得过高动量分。"
                if momentum_vol_available
                else "需要至少 126 根 K 线以估计更稳定的波动调整动量。"
            ),
            available=momentum_vol_available,
        ),
        _factor_signal(
            "momentum_12_1",
            "12-1月动量",
            "动量",
            momentum_12_1,
            momentum_12_1 * 220,
            (
                "跳过最近 1 个月后计算前 12 个月动量。"
                if momentum_12_1_available
                else "需要至少 253 根 K 线，当前明确标记为数据不足。"
            ),
            available=momentum_12_1_available,
        ),
        _factor_signal(
            "relative_strength",
            "相对强弱",
            "动量",
            relative_strength,
            relative_strength * 500,
            "相对基准的 20 日超额收益；无基准时使用绝对动量。",
        ),
        _factor_signal(
            "breakout_20d",
            "20日突破",
            "突破",
            breakout_position,
            (breakout_position - 0.5) * 180,
            "现价在近 20 日高低区间中的位置。",
        ),
        _factor_signal(
            "high_52w_proximity",
            "52周高点接近度",
            "突破",
            high_52w_proximity,
            (high_52w_proximity + 0.12) * 400,
            (
                "衡量现价距离过去 252 个交易日最高收盘价的幅度。"
                if high_52w_available
                else "需要至少 253 根 K 线，当前明确标记为数据不足。"
            ),
            available=high_52w_available,
        ),
        _factor_signal(
            "rsi_reversal",
            "RSI反转",
            "反转",
            rsi14,
            (50 - rsi14) * 2.4,
            "RSI 超买时偏空、超卖时偏多的反转信号。",
        ),
        _factor_signal(
            "bollinger_reversal",
            "布林反转",
            "反转",
            bollinger_z,
            -bollinger_z * 42,
            "价格偏离 20 日均线的标准差倍数。",
        ),
        _factor_signal(
            "money_flow_reversal",
            "MFI资金反转",
            "资金流",
            money_flow_index,
            (50 - money_flow_index) * 2.0,
            "以典型价格和成交量构造资金流指数，超买偏空、超卖偏多。",
        ),
        _factor_signal(
            "volatility_quality",
            "波动质量",
            "风险",
            realized_volatility,
            (0.55 - volatility_percentile) * 150,
            "近 20 日波动率相对自身历史分位，低波动得分更高。",
        ),
        _factor_signal(
            "parkinson_quality",
            "Parkinson区间波动",
            "风险",
            parkinson_volatility,
            (0.5 - parkinson_percentile) * 160,
            "使用每日最高价和最低价估算区间波动，低历史分位得分更高。",
        ),
        _factor_signal(
            "downside_quality",
            "下行风险质量",
            "风险",
            downside_volatility,
            (0.5 - downside_percentile) * 160,
            "只统计负收益的年化下行波动，低历史分位得分更高。",
        ),
        _factor_signal(
            "expected_shortfall_quality",
            "期望损失质量",
            "风险",
            expected_shortfall_value,
            (0.5 - expected_shortfall_percentile) * 170,
            "统计 63 日最差 5% 收益的平均损失，低历史分位得分更高。",
            available=expected_shortfall_available,
        ),
        _factor_signal(
            "ulcer_quality",
            "持续回撤质量",
            "风险",
            ulcer_value,
            (0.5 - ulcer_percentile) * 165,
            "同时惩罚回撤深度与持续时间，低历史分位得分更高。",
            available=ulcer_available,
        ),
        _factor_signal(
            "drawdown_quality",
            "回撤质量",
            "风险",
            drawdown,
            45 + drawdown * 260,
            "现价相对近一年滚动峰值的回撤。",
        ),
        _factor_signal(
            "volume_confirmation",
            "量能确认",
            "量价",
            volume_ratio,
            np.sign(ret5 or ret20) * (volume_ratio - 0.75) * 80,
            "当前成交量相对 20 日均量，并由短期价格方向确认。",
        ),
        _factor_signal(
            "obv_trend",
            "OBV资金趋势",
            "量价",
            obv_change,
            obv_change * 180,
            "近 20 日能量潮变化，用于观察量价累积方向。",
        ),
        _factor_signal(
            "amihud_liquidity",
            "Amihud流动性",
            "流动性",
            amihud_value,
            (0.5 - amihud_percentile) * 160,
            "单位成交额引发的绝对收益变化；数值和历史分位越低越好。",
            available=amihud_available,
        ),
        _factor_signal(
            "overnight_strength",
            "隔夜收益强度",
            "时段结构",
            overnight_strength,
            overnight_strength * 500,
            "近 20 日收盘到次日开盘收益的复合强度。",
            available=overnight_available,
        ),
        _factor_signal(
            "intraday_strength",
            "日内收益强度",
            "时段结构",
            intraday_strength,
            intraday_strength * 500,
            "近 20 日开盘到收盘收益的复合强度。",
            available=intraday_available,
        ),
        _factor_signal(
            "gap_risk_quality",
            "隔夜跳空风险",
            "风险",
            gap_risk,
            (0.5 - gap_risk_percentile) * 160,
            "近 20 日隔夜收益波动相对自身历史分位，低跳空风险得分更高。",
            available=gap_risk_available,
        ),
    )
    return signals


def detect_market_regime(bars: pd.DataFrame) -> MarketRegime:
    frame = _validated_bars(bars)
    close = frame["Close"]
    returns = close.pct_change()
    latest = _latest(close)
    sma20 = _latest(close.rolling(20).mean(), latest)
    sma60 = _latest(close.rolling(60).mean(), latest)
    ret20 = _latest(close.pct_change(20))
    vol20 = _latest(returns.rolling(20).std(ddof=0)) * math.sqrt(252)
    vol_history = returns.rolling(20).std(ddof=0).tail(120).dropna() * math.sqrt(252)
    vol_percentile = float((vol_history <= vol20).mean()) if len(vol_history) else 0.5

    if vol20 >= 0.48 or vol_percentile >= 0.85:
        key = "high_volatility"
        name = "高波动"
        description = "短期波动处于高位，策略仓位需要主动收缩。"
    elif latest > sma20 > sma60 and ret20 > 0:
        key = "bull_trend"
        name = "多头趋势"
        description = "价格位于中短期均线上方，趋势与收益方向一致。"
    elif latest < sma20 < sma60 and ret20 < 0:
        key = "bear_trend"
        name = "空头趋势"
        description = "价格位于中短期均线下方，下行趋势占优。"
    elif latest > sma60 and ret20 > 0:
        key = "recovery"
        name = "修复阶段"
        description = "中期结构改善，但趋势排列尚未完全确认。"
    else:
        key = "range_bound"
        name = "区间震荡"
        description = "趋势一致性较弱，均值回归策略权重上升。"

    trend = "up" if latest > sma20 > sma60 else "down" if latest < sma20 < sma60 else "flat"
    volatility = "high" if vol_percentile >= 0.75 else "low" if vol_percentile <= 0.3 else "normal"
    separation = abs(latest / max(sma60, 1e-12) - 1)
    confidence = int(max(45, min(92, 52 + separation * 520 + abs(ret20) * 120)))
    return MarketRegime(
        key=key,
        name=name,
        trend=trend,
        volatility=volatility,
        confidence=confidence,
        description=description,
    )


def _strategy_weights(regime: MarketRegime) -> dict[str, float]:
    weights = {
        "trend": 0.25,
        "momentum": 0.24,
        "breakout": 0.18,
        "mean_reversion": 0.18,
        "volume_price": 0.15,
    }
    if regime.key == "bull_trend":
        weights.update(trend=0.32, momentum=0.28, breakout=0.22, mean_reversion=0.07, volume_price=0.11)
    elif regime.key == "bear_trend":
        weights.update(trend=0.34, momentum=0.27, breakout=0.16, mean_reversion=0.08, volume_price=0.15)
    elif regime.key == "range_bound":
        weights.update(trend=0.14, momentum=0.15, breakout=0.12, mean_reversion=0.39, volume_price=0.20)
    elif regime.key == "high_volatility":
        weights.update(trend=0.27, momentum=0.16, breakout=0.13, mean_reversion=0.17, volume_price=0.27)
    return weights


def build_strategy_signals(
    factors: tuple[FactorSignal, ...],
    regime: MarketRegime,
) -> tuple[StrategySignal, ...]:
    scores = {factor.key: factor.score for factor in factors}
    availability = {factor.key: factor.available for factor in factors}

    def blend(components: tuple[tuple[str, float], ...]) -> float:
        usable = [
            (key, weight)
            for key, weight in components
            if availability.get(key, True)
        ]
        weight_sum = sum(weight for _, weight in usable)
        if weight_sum <= 0:
            return 0.0
        return sum(scores[key] * weight for key, weight in usable) / weight_sum

    raw = {
        "trend": blend(
            (
                ("trend_alignment", 0.34),
                ("trend_slope", 0.22),
                ("directional_movement", 0.28),
                ("trend_efficiency", 0.16),
            )
        ),
        "momentum": blend(
            (
                ("momentum_5d", 0.08),
                ("momentum_20d", 0.20),
                ("momentum_60d", 0.14),
                ("momentum_12_1", 0.16),
                ("momentum_vol_adjusted", 0.16),
                ("relative_strength", 0.10),
                ("high_52w_proximity", 0.08),
                ("overnight_strength", 0.04),
                ("intraday_strength", 0.04),
            )
        ),
        "breakout": blend(
            (
                ("breakout_20d", 0.50),
                ("high_52w_proximity", 0.25),
                ("volume_confirmation", 0.25),
            )
        ),
        "mean_reversion": blend(
            (
                ("rsi_reversal", 0.40),
                ("bollinger_reversal", 0.35),
                ("money_flow_reversal", 0.25),
            )
        ),
        "volume_price": blend(
            (
                ("volume_confirmation", 0.35),
                ("obv_trend", 0.30),
                ("amihud_liquidity", 0.20),
                ("money_flow_reversal", 0.15),
            )
        ),
    }
    weights = _strategy_weights(regime)
    rationales = {
        "trend": "融合均线排列、对数斜率、ADX 与路径效率。",
        "momentum": "融合多周期、波动调整、52 周高点及相对强弱动量。",
        "breakout": "观察 20 日区间、52 周高点位置与成交量确认。",
        "mean_reversion": "根据 RSI、布林偏离与 MFI 寻找反转条件。",
        "volume_price": "用量能、OBV、MFI 与流动性共同验证价格信号。",
    }
    return tuple(
        StrategySignal(
            key=key,
            name=STRATEGY_NAMES[key],
            score=round(_clip(score), 2),
            weight=weights[key],
            action=_action(score),
            rationale=rationales[key],
        )
        for key, score in raw.items()
    )


def mine_time_series_factors(
    bars: pd.DataFrame,
    *,
    horizon_days: int = 5,
    minimum_observations: int = 40,
) -> tuple[FactorMiningResult, ...]:
    if horizon_days < 1:
        raise ValueError("因子检验周期必须为正整数。")
    frame = _validated_bars(bars)
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    open_price = frame["Open"]
    volume = frame["Volume"].clip(lower=0)
    returns = close.pct_change()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    std20 = close.rolling(20).std(ddof=0)
    range_variance = np.log(
        high.where(high > 0).div(low.where(low > 0))
    ).pow(2).div(4 * math.log(2))
    downside_volatility = returns.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
    dollar_volume = close.mul(volume).where(volume > 0)
    amihud = returns.abs().div(dollar_volume.replace(0, np.nan)).rolling(20).mean()
    overnight = _rolling_compound(open_price.div(close.shift(1)).sub(1))
    intraday = _rolling_compound(close.div(open_price.replace(0, np.nan)).sub(1))
    volatility_63 = returns.rolling(63).std(ddof=0).mul(math.sqrt(252))
    momentum_vol_adjusted = close.pct_change(63).div(
        volatility_63.replace(0, np.nan)
    )
    trend_efficiency = close.diff(20).div(
        close.diff().abs().rolling(20).sum().replace(0, np.nan)
    )
    high_52w_proximity = close.div(close.rolling(252).max()).sub(1)

    def rolling_expected_shortfall(window: np.ndarray) -> float:
        finite = window[np.isfinite(window)]
        if len(finite) < 20:
            return np.nan
        threshold = float(np.quantile(finite, 0.05))
        tail = finite[finite <= threshold]
        return float(-tail.mean()) if len(tail) else np.nan

    expected_shortfall = returns.rolling(63).apply(
        rolling_expected_shortfall,
        raw=True,
    )
    drawdown_63 = close.div(close.rolling(63).max()).sub(1)
    ulcer_index = drawdown_63.pow(2).rolling(63).mean().pow(0.5)
    gap_risk = open_price.div(close.shift(1)).sub(1).rolling(20).std(ddof=0)
    factors = {
        "momentum_20d": ("20日动量", close.pct_change(20)),
        "momentum_12_1": (
            "12-1月动量",
            close.shift(21).div(close.shift(252)).sub(1),
        ),
        "trend_alignment": ("均线排列", close.div(sma60).sub(1)),
        "directional_movement": (
            "ADX方向强度",
            _directional_strength(high, low, close),
        ),
        "bollinger_reversal": ("布林反转", -close.sub(sma20).div(std20.replace(0, np.nan))),
        "money_flow_reversal": (
            "MFI资金反转",
            50 - _money_flow_index(high, low, close, volume),
        ),
        "volatility_quality": ("低波动", -returns.rolling(20).std(ddof=0)),
        "parkinson_quality": (
            "Parkinson区间波动",
            -range_variance.rolling(20).mean().pow(0.5),
        ),
        "downside_quality": ("下行风险质量", -downside_volatility),
        "volume_confirmation": (
            "量价确认",
            close.pct_change(5) * volume.div(volume.rolling(20).mean()),
        ),
        "amihud_liquidity": ("Amihud流动性", -amihud),
        "overnight_strength": ("隔夜收益强度", overnight),
        "intraday_strength": ("日内收益强度", intraday),
        "breakout_20d": (
            "20日突破",
            close.div(close.rolling(20).max()).sub(1),
        ),
        "momentum_vol_adjusted": ("波动调整动量", momentum_vol_adjusted),
        "trend_efficiency": ("趋势效率", trend_efficiency),
        "high_52w_proximity": ("52周高点接近度", high_52w_proximity),
        "expected_shortfall_quality": ("期望损失质量", -expected_shortfall),
        "ulcer_quality": ("持续回撤质量", -ulcer_index),
        "gap_risk_quality": ("隔夜跳空风险", -gap_risk),
    }
    forward = close.shift(-horizon_days).div(close).sub(1).rename("forward")
    results: list[FactorMiningResult] = []
    for key, (name, series) in factors.items():
        sample = pd.concat([series.rename("factor"), forward], axis=1).dropna()
        if len(sample) < minimum_observations:
            continue
        ic = _rank_correlation(sample["factor"], sample["forward"])
        denominator = max(1e-9, 1 - ic * ic)
        t_statistic = ic * math.sqrt(max(len(sample) - 2, 1) / denominator)
        nonzero = sample[(sample["factor"] != 0) & (sample["forward"] != 0)]
        win_rate = (
            float((np.sign(nonzero["factor"]) == np.sign(nonzero["forward"])).mean())
            if len(nonzero)
            else 0.5
        )
        fold_ics: list[float] = []
        fold_size = math.ceil(len(sample) / 3)
        for fold_index in range(3):
            fold = sample.iloc[
                fold_index * fold_size : min((fold_index + 1) * fold_size, len(sample))
            ]
            if len(fold) < 20:
                continue
            fold_ics.append(_rank_correlation(fold["factor"], fold["forward"]))
        meaningful_signs = [np.sign(value) for value in fold_ics if abs(value) >= 0.01]
        sign_consistency = (
            max(0.0, float(np.mean(meaningful_signs))) if meaningful_signs else 0.0
        )
        magnitude = min(1.0, max(ic, 0.0) / 0.08)
        dispersion = max(
            0.0,
            1.0 - (float(np.std(fold_ics)) / 0.12 if fold_ics else 1.0),
        )
        sample_strength = min(1.0, len(sample) / 252)
        stability_score = 100 * (
            0.45 * sign_consistency
            + 0.30 * magnitude
            + 0.15 * dispersion
            + 0.10 * sample_strength
        )
        if len(fold_ics) < 2:
            stability_score *= 0.6
        results.append(
            FactorMiningResult(
                key=key,
                name=name,
                information_coefficient=round(ic, 4),
                t_statistic=round(float(t_statistic), 3),
                directional_win_rate=round(win_rate, 4),
                observations=len(sample),
                horizon_days=horizon_days,
                category=str(FACTOR_REFERENCE_CATALOG.get(key, {}).get("category", "")),
                reference_title=str(
                    FACTOR_REFERENCE_CATALOG.get(key, {}).get("reference_title", "")
                ),
                reference_url=str(
                    FACTOR_REFERENCE_CATALOG.get(key, {}).get("reference_url", "")
                ),
                fold_information_coefficients=tuple(
                    round(value, 4) for value in fold_ics
                ),
                stability_score=round(stability_score, 1),
            )
        )
    return tuple(sorted(results, key=lambda item: abs(item.information_coefficient), reverse=True))


def build_factor_history(
    bars: pd.DataFrame,
    *,
    periods: int = 30,
) -> list[dict[str, Any]]:
    frame = _validated_bars(bars)
    start = max(80, len(frame) - max(1, periods))
    history: list[dict[str, Any]] = []
    for position in range(start, len(frame)):
        partial = frame.iloc[: position + 1]
        signals = calculate_factor_signals(partial)
        by_key = {item.key: item for item in signals}

        def category_score(keys: tuple[str, ...]) -> float:
            usable = [by_key[key].score for key in keys if by_key[key].available]
            return round(float(np.mean(usable)), 2) if usable else 0.0

        timestamp = pd.Timestamp(partial.index[-1])
        history.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "trend": category_score(
                    (
                        "trend_alignment",
                        "trend_slope",
                        "directional_movement",
                        "trend_efficiency",
                    )
                ),
                "momentum": category_score(
                    (
                        "momentum_5d",
                        "momentum_20d",
                        "momentum_60d",
                        "momentum_12_1",
                        "relative_strength",
                        "breakout_20d",
                        "momentum_vol_adjusted",
                        "high_52w_proximity",
                    )
                ),
                "reversal": category_score(
                    ("rsi_reversal", "bollinger_reversal")
                ),
                "risk": category_score(
                    (
                        "volatility_quality",
                        "parkinson_quality",
                        "downside_quality",
                        "drawdown_quality",
                        "expected_shortfall_quality",
                        "ulcer_quality",
                        "gap_risk_quality",
                    )
                ),
                "liquidity": category_score(("amihud_liquidity",)),
                "flow": category_score(
                    ("volume_confirmation", "obv_trend", "money_flow_reversal")
                ),
                "session": category_score(
                    ("overnight_strength", "intraday_strength")
                ),
            }
        )
    return history


def analyze_composite(
    bars: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> CompositeResearch:
    frame = _validated_bars(bars)
    factors = calculate_factor_signals(frame, benchmark)
    regime = detect_market_regime(frame)
    strategies = build_strategy_signals(factors, regime)
    mining = mine_time_series_factors(frame)
    raw_directional_score = _clip(
        sum(item.score * item.weight for item in strategies)
    )
    gross_direction = sum(abs(item.score) * item.weight for item in strategies)
    agreement = (
        min(1.0, abs(raw_directional_score) / gross_direction)
        if gross_direction > 1e-9
        else 0.5
    )

    risk_factors = {item.key: item for item in factors}
    volatility = max(
        risk_factors["volatility_quality"].value,
        risk_factors["parkinson_quality"].value,
        0.05,
    )
    downside_volatility = max(risk_factors["downside_quality"].value, 0.0)
    drawdown = min(risk_factors["drawdown_quality"].value, 0.0)
    expected_shortfall = max(
        risk_factors["expected_shortfall_quality"].value,
        0.0,
    )
    ulcer_index = max(risk_factors["ulcer_quality"].value, 0.0)
    gap_risk = max(risk_factors["gap_risk_quality"].value, 0.0)
    risk_penalty = max(0.0, -risk_factors["volatility_quality"].score) * 0.10
    risk_penalty += max(0.0, -risk_factors["parkinson_quality"].score) * 0.05
    risk_penalty += max(0.0, -risk_factors["downside_quality"].score) * 0.08
    risk_penalty += max(0.0, -risk_factors["drawdown_quality"].score) * 0.08
    risk_penalty += max(0.0, -risk_factors["amihud_liquidity"].score) * 0.04
    risk_penalty += max(0.0, -risk_factors["expected_shortfall_quality"].score) * 0.06
    risk_penalty += max(0.0, -risk_factors["ulcer_quality"].score) * 0.05
    risk_penalty += max(0.0, -risk_factors["gap_risk_quality"].score) * 0.04
    stability_values = [
        item.stability_score for item in mining if item.observations >= 80
    ]
    factor_stability = (
        float(np.median(stability_values)) / 100 if stability_values else 0.35
    )
    factor_stability = max(0.0, min(1.0, factor_stability))
    total_factor_weight = sum(item.weight for item in factors)
    factor_coverage = (
        sum(item.weight for item in factors if item.available) / total_factor_weight
        if total_factor_weight
        else 0.0
    )
    regime_reliability = max(0.0, min(1.0, regime.confidence / 100))
    evidence_reliability = (
        0.35 * agreement
        + 0.25 * factor_stability
        + 0.25 * factor_coverage
        + 0.15 * regime_reliability
    )
    calibration_factor = 0.50 + 0.50 * evidence_reliability
    directional_score = _clip(raw_directional_score * calibration_factor)
    signal_clarity = min(1.0, abs(raw_directional_score) / 40.0)
    risk_reliability = max(0.55, 1.0 - min(risk_penalty, 45.0) / 100.0)
    confidence = round(
        max(
            18.0,
            min(
                90.0,
                100
                * evidence_reliability
                * (0.68 + 0.32 * signal_clarity)
                * risk_reliability,
            ),
        )
    )

    if directional_score <= 0:
        base_position = 0.0 if directional_score <= -25 else 0.08
    else:
        base_position = min(0.88, 0.12 + directional_score / 100 * 0.82)
    volatility_scale = max(0.25, min(1.0, 0.28 / volatility))
    downside_scale = max(
        0.35,
        min(1.0, 0.22 / max(downside_volatility, 0.05)),
    )
    drawdown_scale = max(0.35, min(1.0, 1 + drawdown * 1.25))
    liquidity_scale = max(
        0.55,
        min(1.0, 1 + risk_factors["amihud_liquidity"].score / 180),
    )
    regime_scale = 0.65 if regime.key == "high_volatility" else 1.0
    tail_scale = max(0.40, min(1.0, 0.32 / max(expected_shortfall, 0.05)))
    ulcer_scale = max(0.45, min(1.0, 0.12 / max(ulcer_index, 0.02)))
    gap_scale = max(0.65, min(1.0, 0.18 / max(gap_risk, 0.03)))
    target_position = round(
        base_position
        * volatility_scale
        * downside_scale
        * drawdown_scale
        * liquidity_scale
        * regime_scale
        * tail_scale
        * ulcer_scale
        * gap_scale,
        4,
    )

    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"].sub(frame["Low"]),
            frame["High"].sub(previous_close).abs(),
            frame["Low"].sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = _latest(true_range.rolling(14).mean()) / _latest(frame["Close"])
    stop_loss = round(max(0.035, min(0.18, atr_pct * 2.2)), 4)
    take_profit = round(min(0.38, stop_loss * (2.0 if confidence >= 60 else 1.7)), 4)

    liquidity_score = risk_factors["amihud_liquidity"].score
    if (
        volatility >= 0.5
        or downside_volatility >= 0.35
        or drawdown <= -0.4
        or liquidity_score <= -65
        or expected_shortfall >= 0.55
        or ulcer_index >= 0.25
        or gap_risk >= 0.45
    ):
        risk_level = "高"
    elif (
        volatility >= 0.3
        or downside_volatility >= 0.2
        or drawdown <= -0.22
        or liquidity_score <= -35
        or expected_shortfall >= 0.35
        or ulcer_index >= 0.12
        or gap_risk >= 0.25
    ):
        risk_level = "中"
    else:
        risk_level = "低"

    action = _action(directional_score)
    label = _action_label(directional_score)
    rounded_direction = round(directional_score, 2)
    rating_score = round(_rating_score(rounded_direction), 2)
    summary = (
        f"当前处于{regime.name}，研判评分 {rating_score:.1f}/100"
        f"（{_direction_phrase(directional_score)}，方向强度 {abs(directional_score):.1f}），"
        f"一致度 {agreement:.0%}，历史分段稳定度 {factor_stability:.0%}；"
        f"建议目标仓位 {target_position:.0%}，"
        f"风险等级{risk_level}。"
    )
    return CompositeResearch(
        score=rating_score,
        directional_score=rounded_direction,
        raw_directional_score=round(raw_directional_score, 2),
        signal_strength=round(abs(directional_score), 2),
        agreement=round(agreement, 6),
        factor_stability=round(factor_stability, 6),
        factor_coverage=round(factor_coverage, 6),
        calibration_factor=round(calibration_factor, 6),
        action=action,
        action_label=label,
        confidence=confidence,
        target_position=target_position,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        risk_level=risk_level,
        regime=regime,
        factors=factors,
        strategies=strategies,
        factor_mining=mining,
        summary=summary,
    )
