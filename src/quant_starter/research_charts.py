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
import numpy as np

from quant_starter.agent_workflow import ResearchResult


def save_research_overview(result: ResearchResult, run_dir: str | Path) -> Path:
    """Create a compact evidence chart that matches the saved agent reports."""

    output_path = Path(run_dir) / "research_overview.png"
    bars = result.context.bars.tail(260).copy()
    close = bars["Close"].astype(float)
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    rsi = rsi.where(loss != 0, 100.0).where(gain != 0, 0.0)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50.0)

    action_colors = {"BUY": "#16794A", "HOLD": "#9A6700", "SELL": "#B42335"}
    action_color = action_colors.get(result.decision.action, "#273043")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0, 0.8]},
    )
    price_ax, rsi_ax, volume_ax = axes
    price_ax.plot(close.index, close.to_numpy(), color="#273043", linewidth=1.35, label="Close")
    price_ax.plot(sma5.index, sma5.to_numpy(), color="#168AAD", linewidth=1.2, label="SMA5")
    price_ax.plot(sma20.index, sma20.to_numpy(), color="#F18F01", linewidth=1.2, label="SMA20")
    price_ax.scatter(
        [close.index[-1]],
        [close.iloc[-1]],
        color=action_color,
        s=76,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label=f"Decision: {result.decision.action}",
    )
    price_ax.set_title(
        f"{result.context.symbol} Research Evidence | "
        f"{result.decision.action} | Confidence {result.decision.confidence}%"
    )
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.2)
    price_ax.legend(loc="best", ncol=4)

    rsi_ax.plot(rsi.index, rsi.to_numpy(), color="#6B5B95", linewidth=1.2)
    rsi_ax.axhline(70, color="#C73E4D", linewidth=0.9, linestyle="--")
    rsi_ax.axhline(30, color="#2A9D5B", linewidth=0.9, linestyle="--")
    rsi_ax.fill_between(rsi.index, 30, 70, color="#D9E2E8", alpha=0.22)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_ylabel("RSI14")
    rsi_ax.grid(True, alpha=0.18)

    volume_ax.bar(
        bars.index,
        bars["Volume"].astype(float),
        color="#7A7D7D",
        width=1.0,
        alpha=0.7,
    )
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, axis="y", alpha=0.18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=155)
    plt.close(fig)
    return output_path
