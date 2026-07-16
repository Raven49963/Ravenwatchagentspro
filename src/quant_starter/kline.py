from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from .data import OHLCV_COLUMNS


UP_COLOR = "#C43D4D"
DOWN_COLOR = "#16856B"
WICK_COLOR = "#5E6872"


def _format_volume(value: float, _position: float) -> str:
    absolute = abs(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return f"{value:.0f}"


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        raise ValueError("K-line data is empty.")
    missing = [column for column in OHLCV_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError("K-line data is missing: " + ", ".join(missing))
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("K-line dates must be sorted and unique.")

    numeric = bars.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("K-line data contains missing or non-numeric values.")
    if (numeric.loc[:, ["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("K-line OHLC prices must be positive.")
    if (numeric["High"] < numeric[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("K-line High is below Open or Close.")
    if (numeric["Low"] > numeric[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("K-line Low is above Open or Close.")
    if (numeric["Volume"] < 0).any():
        raise ValueError("K-line Volume cannot be negative.")
    return numeric


def build_kline_figure(
    bars: pd.DataFrame,
    symbol: str,
    window: int | None = 120,
) -> Figure:
    """Build an OHLC candlestick chart with moving averages and volume."""

    numeric = _validated_bars(bars)
    if window is not None:
        if not 20 <= window <= 2000:
            raise ValueError("K-line window must be between 20 and 2000 rows.")
        view = numeric.tail(window).copy()
    else:
        view = numeric.copy()
    if len(view) < 2:
        raise ValueError("K-line chart requires at least two rows.")

    moving_averages = {
        period: numeric["Close"].rolling(period).mean().reindex(view.index)
        for period in (5, 20, 60)
    }
    x = np.arange(len(view), dtype=float)
    opens = view["Open"].to_numpy(dtype=float)
    highs = view["High"].to_numpy(dtype=float)
    lows = view["Low"].to_numpy(dtype=float)
    closes = view["Close"].to_numpy(dtype=float)
    volumes = view["Volume"].to_numpy(dtype=float)
    colors = np.where(closes >= opens, UP_COLOR, DOWN_COLOR)

    figure = Figure(figsize=(9.2, 5.4), dpi=100, facecolor="#FFFFFF")
    grid = figure.add_gridspec(4, 1, hspace=0.05)
    price_ax = figure.add_subplot(grid[:3, 0])
    volume_ax = figure.add_subplot(grid[3, 0], sharex=price_ax)

    visible_range = max(float(highs.max() - lows.min()), float(closes.mean()) * 0.001)
    minimum_body = visible_range * 0.0015
    candle_width = 0.62
    for position, open_price, high, low, close, color in zip(
        x, opens, highs, lows, closes, colors, strict=True
    ):
        price_ax.vlines(position, low, high, color=WICK_COLOR, linewidth=0.8, zorder=1)
        body_height = abs(close - open_price)
        if body_height < minimum_body:
            body_height = minimum_body
            body_bottom = (open_price + close) / 2.0 - body_height / 2.0
        else:
            body_bottom = min(open_price, close)
        price_ax.add_patch(
            Rectangle(
                (position - candle_width / 2.0, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
                zorder=2,
            )
        )

    ma_colors = {5: "#D48A13", 20: "#3478B8", 60: "#7A5AA6"}
    for period, series in moving_averages.items():
        price_ax.plot(
            x,
            series.to_numpy(dtype=float),
            color=ma_colors[period],
            linewidth=1.05,
            label=f"MA{period}",
        )

    previous_close = closes[-2]
    daily_change = closes[-1] / previous_close - 1.0 if previous_close else 0.0
    price_ax.set_title(
        f"{symbol.upper()} | {view.index[-1].date()} | "
        f"Close {closes[-1]:,.2f} | {daily_change:+.2%}",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    price_ax.set_ylabel("Price")
    price_ax.set_xlim(-1, len(view))
    price_ax.grid(True, color="#E7EBEF", linewidth=0.65, alpha=0.85)
    price_ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    price_ax.tick_params(axis="x", labelbottom=False)
    for spine in price_ax.spines.values():
        spine.set_color("#DCE2E7")

    volume_ax.bar(x, volumes, width=candle_width, color=colors, alpha=0.72)
    volume_ax.set_ylabel("Volume")
    volume_ax.yaxis.set_major_formatter(FuncFormatter(_format_volume))
    volume_ax.grid(True, axis="y", color="#E7EBEF", linewidth=0.6, alpha=0.75)
    for spine in volume_ax.spines.values():
        spine.set_color("#DCE2E7")

    single_year = view.index[0].year == view.index[-1].year
    tick_count = min(6 if single_year else 5, len(view))
    tick_positions = np.unique(
        np.linspace(0, len(view) - 1, num=tick_count, dtype=int)
    )
    volume_ax.set_xticks(tick_positions)
    volume_ax.set_xticklabels(
        [
            view.index[position].strftime("%m-%d" if single_year else "%Y-%m-%d")
            for position in tick_positions
        ],
        rotation=0 if single_year else 35,
        ha="center",
        fontsize=8,
    )
    tick_labels = volume_ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
    figure.subplots_adjust(left=0.14, right=0.985, top=0.92, bottom=0.18)
    return figure


def save_kline_chart(
    bars: pd.DataFrame,
    symbol: str,
    output_path: str | Path,
    window: int | None = 250,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_kline_figure(bars, symbol, window)
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    figure.clear()
    return path
