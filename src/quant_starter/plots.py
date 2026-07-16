from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "RavenWatchAgentsCN-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from quant_starter.backtest import BacktestResult
from quant_starter.metrics import drawdown


def save_report_charts(result: BacktestResult, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / "equity_vs_benchmark.png"
    drawdown_path = output_dir / "drawdown.png"
    allocation_path = output_dir / "allocation.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity.plot(ax=ax, label="Strategy", linewidth=2)
    result.benchmark_equity.plot(
        ax=ax, label=result.benchmark_name, linewidth=1.5
    )
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(equity_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    drawdown(result.equity).plot(ax=ax, color="#b33a3a", linewidth=1.5)
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(drawdown_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    result.target_weights.plot.area(ax=ax, stacked=True, linewidth=0, alpha=0.85)
    ax.set_title("Target Allocation")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(allocation_path, dpi=160)
    plt.close(fig)

    return [equity_path, drawdown_path, allocation_path]


def save_moving_average_chart(
    result: BacktestResult,
    signals,
    ticker: str,
    output_dir: Path,
) -> Path:
    """Save price, crossover markers, and the equity curve in one figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "moving_average_backtest.png"
    sma_columns = [column for column in signals.columns if column.startswith("SMA")]
    if len(sma_columns) != 2:
        raise ValueError("The signal table must contain two SMA columns.")

    buys = signals[signals["Signal"] == 1]
    sells = signals[signals["Signal"] == -1]

    fig, (price_ax, equity_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    signals["Close"].plot(
        ax=price_ax, color="#273043", linewidth=1.2, label=f"{ticker} Close"
    )
    signals[sma_columns[0]].plot(
        ax=price_ax, color="#168AAD", linewidth=1.3, label=sma_columns[0]
    )
    signals[sma_columns[1]].plot(
        ax=price_ax, color="#F18F01", linewidth=1.3, label=sma_columns[1]
    )
    price_ax.scatter(
        buys.index,
        buys["Close"],
        marker="^",
        s=68,
        color="#2A9D5B",
        edgecolors="white",
        linewidths=0.6,
        label="Buy",
        zorder=4,
    )
    price_ax.scatter(
        sells.index,
        sells["Close"],
        marker="v",
        s=68,
        color="#C73E4D",
        edgecolors="white",
        linewidths=0.6,
        label="Sell",
        zorder=4,
    )
    price_ax.set_title("Double Moving-Average Signals")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.22)
    price_ax.legend(loc="best", ncol=3)

    result.equity.plot(
        ax=equity_ax, color="#273043", linewidth=1.8, label="Strategy"
    )
    result.benchmark_equity.plot(
        ax=equity_ax,
        color="#7A7D7D",
        linewidth=1.2,
        linestyle="--",
        label=result.benchmark_name,
    )
    equity_ax.set_title("Portfolio Equity")
    equity_ax.set_ylabel("Value")
    equity_ax.grid(True, alpha=0.22)
    equity_ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path
