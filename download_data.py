from __future__ import annotations

"""下载并检查一只股票的日线 OHLCV 数据。

默认示例会下载 AAPL 最近两年的数据。这个脚本刻意保持简单，适合先单独
理解“获取数据 -> 清洗字段 -> 保存 CSV”，再进入策略和回测模块。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_starter.data import (
    DemoMarketConfig,
    fetch_a_share_ohlcv,
    fetch_nasdaq_ohlcv,
    generate_demo_ohlcv,
    normalize_a_share_symbol,
)


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.today().normalize()
    two_years_ago = today - pd.DateOffset(years=2)

    parser = argparse.ArgumentParser(description="下载股票日线 OHLCV 数据")
    parser.add_argument(
        "--market",
        choices=("demo", "a-share", "nasdaq"),
        default="nasdaq",
        help="数据市场：模拟数据、A 股或纳斯达克/美股。",
    )
    parser.add_argument("--symbol", default="AAPL", help="股票代码，例如 AAPL 或 600519。")
    parser.add_argument(
        "--start",
        default=two_years_ago.strftime("%Y-%m-%d"),
        help="开始日期，默认是两年前。",
    )
    parser.add_argument(
        "--end",
        default=today.strftime("%Y-%m-%d"),
        help="结束日期，默认是今天。",
    )
    parser.add_argument(
        "--adjust",
        choices=("qfq", "hfq", "none"),
        default="qfq",
        help="A 股复权方式。",
    )
    parser.add_argument(
        "--no-auto-adjust",
        action="store_false",
        dest="auto_adjust",
        help="美股不自动调整 OHLC 价格。",
    )
    parser.set_defaults(auto_adjust=True)
    parser.add_argument("--output", default=None, help="输出 CSV 路径。")
    return parser.parse_args()


def download(args: argparse.Namespace) -> tuple[str, pd.DataFrame]:
    """根据市场选择对应接口，并返回标准化后的五价量数据。"""

    if args.market == "demo":
        symbol = args.symbol.strip().upper()
        if symbol == "AAPL":
            symbol = "ALPHA"
        config = DemoMarketConfig(start=args.start, end=args.end)
        return symbol, generate_demo_ohlcv(symbol, config)

    if args.market == "a-share":
        symbol = normalize_a_share_symbol(args.symbol)
        bars = fetch_a_share_ohlcv(
            symbol=symbol,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
        )
        return symbol, bars

    symbol = args.symbol.strip().upper()
    bars = fetch_nasdaq_ohlcv(
        symbol=symbol,
        start=args.start,
        end=args.end,
        auto_adjust=args.auto_adjust,
    )
    return symbol, bars


def main() -> None:
    args = parse_args()
    symbol, bars = download(args)

    # 不指定路径时，数据统一放到项目 data 目录，方便后续策略脚本读取。
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / f"{symbol}_ohlcv.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig 让 CSV 在中文版 Excel 中直接打开时也能正确识别编码。
    bars.to_csv(output_path, encoding="utf-8-sig")

    print(f"\n下载完成：{symbol}")
    print(f"日期范围：{bars.index.min().date()} 至 {bars.index.max().date()}")
    print(f"数据行数：{len(bars):,}")
    print(f"保存位置：{output_path.resolve()}")
    print("\n前 5 行数据：")
    print(bars.head().to_string())


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
