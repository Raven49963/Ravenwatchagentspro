from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from quant_starter.backtest import BacktestConfig, run_backtest
from quant_starter.data import (
    DemoMarketConfig,
    fetch_a_share_prices,
    fetch_nasdaq_prices,
    generate_demo_prices,
    load_prices_csv,
    normalize_a_share_symbol,
    parse_symbols,
)
from quant_starter.metrics import format_metrics, summarize_performance
from quant_starter.optimization import (
    HoldoutConfig,
    HoldoutValidationResult,
    optimize_risk_managed_momentum,
)
from quant_starter.plots import save_moving_average_chart, save_report_charts
from quant_starter.strategies import (
    RiskManagedMomentumConfig,
    TacticalGrowthConfig,
    momentum_rotation,
    moving_average_crossover,
    moving_average_signals,
    risk_managed_momentum,
    tactical_growth_allocation,
)
from quant_starter.validation import (
    TacticalGrowthValidationResult,
    validate_tactical_growth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a beginner-friendly stock quant backtest."
    )
    parser.add_argument(
        "--strategy",
        choices=["tactical-growth", "adaptive", "risk-momentum", "momentum", "ma"],
        default="adaptive",
        help="tactical-growth = fixed Nasdaq leveraged allocation with chronological validation; adaptive = train/holdout parameter selection; risk-momentum = volatility-aware momentum; momentum = top-N rotation; ma = moving-average trend.",
    )
    parser.add_argument(
        "--source",
        choices=["demo", "csv", "a-share", "nasdaq"],
        default="demo",
        help="Market data source.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Demo data random seed.")
    parser.add_argument(
        "--prices-csv",
        default=None,
        help="CSV price file. Used when --source csv is selected.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols separated by spaces or commas. A-share examples: 600519 000001. Nasdaq examples: AAPL MSFT NVDA.",
    )
    parser.add_argument(
        "--start",
        default="2010-03-01",
        help="Start date for real market data.",
    )
    parser.add_argument(
        "--end",
        default=pd.Timestamp.now().date().isoformat(),
        help="End date for real market data.",
    )
    parser.add_argument(
        "--adjust",
        choices=["qfq", "hfq", "none"],
        default="qfq",
        help="A-share adjustment mode. qfq=front adjusted, hfq=back adjusted, none=unadjusted.",
    )
    parser.add_argument(
        "--auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use yfinance adjusted prices for Nasdaq/US data.",
    )
    parser.add_argument(
        "--initial-cash", type=float, default=100_000.0, help="Starting capital."
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0003,
        help="One-way commission rate. 0.0003 means 3 basis points.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0002,
        help="Approximate one-way slippage rate.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Momentum lookback window in trading days.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=21,
        help="Momentum rebalance interval in trading days.",
    )
    parser.add_argument(
        "--top-n", type=int, default=2, help="Number of winners in momentum rotation."
    )
    parser.add_argument("--skip-recent", type=int, default=21)
    parser.add_argument("--trend-window", type=int, default=100)
    parser.add_argument("--volatility-window", type=int, default=63)
    parser.add_argument("--target-volatility", type=float, default=0.18)
    parser.add_argument("--max-position", type=float, default=0.65)
    parser.add_argument("--target-annual-return", type=float, default=0.20)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.30)
    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker used by the moving-average strategy. Defaults to the first symbol.",
    )
    parser.add_argument("--fast", type=int, default=5, help="Fast moving average.")
    parser.add_argument("--slow", type=int, default=20, help="Slow moving average.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Folder for CSV files and charts.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip PNG chart generation.",
    )
    return parser.parse_args()


def load_market_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.strategy == "tactical-growth" and args.source not in {"nasdaq", "csv"}:
        raise ValueError(
            "tactical-growth requires --source nasdaq or a CSV containing "
            "QQQ, TQQQ, and BIL."
        )
    if args.prices_csv and args.source == "demo":
        return load_prices_csv(Path(args.prices_csv))

    if args.source == "demo":
        return generate_demo_prices(DemoMarketConfig(seed=args.seed))

    if args.source == "csv":
        if not args.prices_csv:
            raise ValueError("Use --prices-csv when --source csv is selected.")
        return load_prices_csv(Path(args.prices_csv))

    if args.source == "a-share":
        symbols = parse_symbols(args.symbols or "600519,000001,300750,000858,601318")
        return fetch_a_share_prices(
            symbols=symbols,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
        )

    if args.source == "nasdaq":
        default_symbols = (
            "QQQ,TQQQ,BIL"
            if args.strategy == "tactical-growth"
            else "AAPL,MSFT,NVDA,AMZN,GOOGL"
        )
        symbols = parse_symbols(args.symbols or default_symbols)
        return fetch_nasdaq_prices(
            symbols=symbols,
            start=args.start,
            end=args.end,
            auto_adjust=args.auto_adjust,
        )

    raise ValueError(f"Unknown data source: {args.source}")


def effective_source(args: argparse.Namespace) -> str:
    if args.prices_csv and args.source == "demo":
        return "csv"
    return args.source


def effective_ticker(args: argparse.Namespace, prices: pd.DataFrame) -> str:
    if not args.ticker:
        return str(prices.columns[0])
    ticker = args.ticker.strip().upper()
    if args.source == "a-share":
        ticker = normalize_a_share_symbol(ticker)
    return ticker


def build_target_weights(args: argparse.Namespace, prices: pd.DataFrame) -> pd.DataFrame:
    if args.strategy == "tactical-growth":
        return tactical_growth_allocation(prices, TacticalGrowthConfig())
    if args.strategy == "risk-momentum":
        return risk_managed_momentum(
            prices,
            RiskManagedMomentumConfig(
                lookback=args.lookback,
                skip_recent=args.skip_recent,
                trend_window=args.trend_window,
                volatility_window=args.volatility_window,
                rebalance_every=args.rebalance_every,
                top_n=args.top_n,
                target_volatility=args.target_volatility,
                max_position=args.max_position,
            ),
        )
    if args.strategy == "momentum":
        return momentum_rotation(
            prices,
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            top_n=args.top_n,
        )

    return moving_average_crossover(
        prices,
        ticker=effective_ticker(args, prices),
        fast=args.fast,
        slow=args.slow,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_market_prices(args)
    backtest_config = BacktestConfig(
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        benchmark_symbol="QQQ" if args.strategy == "tactical-growth" else None,
    )
    validation = None
    if args.strategy == "tactical-growth":
        validation = validate_tactical_growth(
            prices,
            config=TacticalGrowthConfig(),
            backtest=backtest_config,
            target_annual_return=args.target_annual_return,
        )
        target_weights = validation.target_weights
    elif args.strategy == "adaptive":
        validation = optimize_risk_managed_momentum(
            prices,
            holdout=HoldoutConfig(
                target_annual_return=args.target_annual_return,
                max_drawdown_limit=args.max_drawdown_limit,
            ),
            backtest=backtest_config,
        )
        target_weights = validation.target_weights
    else:
        target_weights = build_target_weights(args, prices)
    ticker = effective_ticker(args, prices) if args.strategy == "ma" else None
    result = run_backtest(
        prices,
        target_weights,
        backtest_config,
    )
    metrics = summarize_performance(
        equity=result.equity,
        returns=result.portfolio_returns,
        turnover=result.turnover,
        target_weights=result.target_weights,
    )

    prices.to_csv(output_dir / "prices.csv")
    result.target_weights.to_csv(output_dir / "target_weights.csv")
    result.equity.rename("equity").to_csv(output_dir / "equity_curve.csv")
    result.turnover.rename("turnover").to_csv(output_dir / "turnover.csv")
    pd.Series(metrics, name="value").to_csv(output_dir / "metrics.csv")
    if isinstance(validation, HoldoutValidationResult):
        validation.leaderboard.to_csv(
            output_dir / "parameter_leaderboard.csv", index=False
        )
        (output_dir / "holdout_validation.json").write_text(
            json.dumps(validation.summary_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif isinstance(validation, TacticalGrowthValidationResult):
        validation.diagnostics.to_csv(
            output_dir / "tactical_growth_diagnostics.csv"
        )
        transitions = validation.diagnostics["RiskOn"].ne(
            validation.diagnostics["RiskOn"].shift()
        )
        transitions.iloc[0] = False
        validation.diagnostics.loc[transitions].to_csv(output_dir / "trades.csv")
        (output_dir / "tactical_growth_validation.json").write_text(
            json.dumps(validation.summary_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    signals = None
    if ticker is not None:
        signals = moving_average_signals(prices, ticker, args.fast, args.slow)
        signals.to_csv(output_dir / "moving_average_signals.csv")
        signals.loc[signals["Signal"] != 0].to_csv(output_dir / "trades.csv")

    chart_paths = []
    if not args.no_charts:
        chart_paths = save_report_charts(result, output_dir)
        if signals is not None and ticker is not None:
            chart_paths.append(
                save_moving_average_chart(result, signals, ticker, output_dir)
            )

    print("\nBacktest complete")
    print(f"Data source: {effective_source(args)}")
    print(f"Strategy: {args.strategy}")
    if ticker is not None:
        print(f"Ticker: {ticker}")
        print(f"Crossovers: {(signals['Signal'] != 0).sum()}")
    print(f"Rows: {len(prices):,}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Output folder: {output_dir.resolve()}")
    print("\nKey metrics")
    print(format_metrics(metrics))
    if isinstance(validation, HoldoutValidationResult):
        print("\nHoldout validation")
        print(f"Split date: {validation.split_date}")
        print(
            "20% target (train / holdout): "
            f"{validation.train_target_met} / {validation.test_target_met}"
        )
        print(f"Holdout drawdown limit met: {validation.test_risk_limit_met}")
        print(f"Selected parameters: {validation.selected}")
        print("\nHoldout metrics")
        print(format_metrics(validation.test_metrics))
    elif isinstance(validation, TacticalGrowthValidationResult):
        print("\nFixed-rule chronological validation")
        print(f"Historical target met: {validation.historical_target_met}")
        print(
            f"Double-cost target met ({validation.stress_one_way_cost:.2%} one-way): "
            f"{validation.cost_stress_target_met}"
        )
        print(
            "Full validation CAGR / drawdown: "
            f"{validation.full_period.metrics['annual_return']:.2%} / "
            f"{validation.full_period.metrics['max_drawdown']:.2%}"
        )
        print(
            "Chronological holdout CAGR / drawdown: "
            f"{validation.holdout_period.metrics['annual_return']:.2%} / "
            f"{validation.holdout_period.metrics['max_drawdown']:.2%}"
        )
        print(
            "Double-cost holdout CAGR: "
            f"{validation.stress_holdout_period.metrics['annual_return']:.2%}"
        )
        print(
            "Four block CAGRs: "
            + " / ".join(
                f"{period.metrics['annual_return']:.2%}"
                for period in validation.chronological_blocks
            )
        )
        print(
            "Warning: TQQQ is a daily leveraged ETF. Historical results do not "
            "guarantee future returns and drawdowns can exceed 50%."
        )

    if chart_paths:
        print("\nCharts")
        for path in chart_paths:
            print(f"- {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
