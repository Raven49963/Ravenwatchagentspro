from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def summarize_performance(
    equity: pd.Series,
    returns: pd.Series,
    turnover: pd.Series | None = None,
    target_weights: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    if equity.empty:
        raise ValueError("equity is empty.")
    if returns.empty:
        raise ValueError("returns is empty.")

    periods = max(len(equity), 1)
    growth = float((1.0 + returns).prod())
    total_return = growth - 1.0
    annual_return = growth ** (TRADING_DAYS / periods) - 1.0
    annual_volatility = returns.std(ddof=0) * math.sqrt(TRADING_DAYS)
    excess_annual_return = annual_return - risk_free_rate
    sharpe = excess_annual_return / annual_volatility if annual_volatility else 0.0
    downside_deviation = float(
        (returns.clip(upper=0.0).pow(2).mean() ** 0.5) * math.sqrt(TRADING_DAYS)
    )
    sortino = (
        excess_annual_return / downside_deviation if downside_deviation else 0.0
    )
    max_drawdown = drawdown(equity).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    win_rate = (returns > 0).mean()
    daily_var_95 = float(returns.quantile(0.05))
    tail = returns.loc[returns <= daily_var_95]
    daily_cvar_95 = float(tail.mean()) if not tail.empty else daily_var_95
    positive_sum = float(returns.loc[returns > 0].sum())
    negative_sum = abs(float(returns.loc[returns < 0].sum()))
    profit_factor = positive_sum / negative_sum if negative_sum else 0.0

    summary = {
        "final_value": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "daily_var_95": daily_var_95,
        "daily_cvar_95": daily_cvar_95,
        "profit_factor": float(profit_factor),
    }

    if turnover is not None:
        summary["avg_daily_turnover"] = float(turnover.mean())
        summary["annualized_turnover"] = float(turnover.mean() * TRADING_DAYS)

    if target_weights is not None:
        summary["average_exposure"] = float(target_weights.sum(axis=1).mean())

    return summary


def format_metrics(metrics: dict[str, float]) -> str:
    labels = {
        "final_value": "最终资金",
        "total_return": "总收益率",
        "annual_return": "年化收益率",
        "annual_volatility": "年化波动率",
        "sharpe": "夏普比率",
        "sortino": "索提诺比率",
        "max_drawdown": "最大回撤",
        "calmar": "卡玛比率",
        "win_rate": "单日胜率",
        "daily_var_95": "单日 VaR(95%)",
        "daily_cvar_95": "单日 CVaR(95%)",
        "profit_factor": "收益因子",
        "avg_daily_turnover": "日均换手率",
        "annualized_turnover": "年化换手率",
        "average_exposure": "平均仓位",
    }
    percent_keys = {
        "total_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "win_rate",
        "daily_var_95",
        "daily_cvar_95",
        "avg_daily_turnover",
        "annualized_turnover",
        "average_exposure",
    }
    lines = []
    for key, value in metrics.items():
        label = labels.get(key, key.replace("_", " "))
        if key == "final_value":
            formatted = f"{value:,.2f}"
        elif key in percent_keys:
            formatted = f"{value:.2%}"
        else:
            formatted = f"{value:.2f}"
        lines.append(f"- {label}: {formatted}")
    return "\n".join(lines)
