from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np
import pandas as pd

from .global_market import fetch_msn_ohlcv
from .runtime import NullTextStream

DEFAULT_TICKERS = ("ALPHA", "BALANCE", "CYCLE", "DEFENSE", "GROWTH", "VALUE")
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TENCENT_KLINE_ENDPOINTS = (
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
)
NASDAQ_API_BASE = "https://api.nasdaq.com/api/quote"
NASDAQ_ETF_SYMBOLS = {
    "DIA",
    "EEM",
    "GLD",
    "IWM",
    "QQQ",
    "SLV",
    "SPY",
    "TLT",
    "VTI",
    "VOO",
    "XLF",
    "XLK",
}
NASDAQ_INDEX_SYMBOLS = {
    "^DJI": "INDU",
    "^GSPC": "SPX",
    "^IXIC": "COMP",
    "^NDX": "NDX",
}
PROVIDER_LABELS = {
    "demo": "模拟行情",
    "csv": "本地 CSV",
    "akshare-eastmoney": "东方财富 / AkShare",
    "tencent-direct": "腾讯证券直连",
    "akshare-tencent": "腾讯证券 / AkShare",
    "yahoo-chart": "Yahoo Chart",
    "yfinance": "yfinance",
    "stooq": "Stooq",
    "nasdaq-api": "Nasdaq 官方公开行情",
    "msn-finance": "Microsoft Finance 全球行情",
    "msn-finance-live": "Microsoft Finance 全球报价",
    "tencent-hk-direct": "腾讯证券港股直连",
    "tencent-realtime": "腾讯证券实时行情",
    "tencent-minute": "腾讯证券分钟行情",
    "nasdaq-public-chart": "Nasdaq 官方盘中行情",
    "yahoo-chart-1m": "Yahoo Finance 分钟行情",
}
PROVIDER_URLS = {
    "demo": "",
    "csv": "",
    "akshare-eastmoney": "https://quote.eastmoney.com/",
    "tencent-direct": "https://gu.qq.com/",
    "akshare-tencent": "https://gu.qq.com/",
    "tencent-hk-direct": "https://gu.qq.com/hk/",
    "yahoo-chart": "https://finance.yahoo.com/",
    "yfinance": "https://finance.yahoo.com/",
    "stooq": "https://stooq.com/",
    "nasdaq-api": "https://www.nasdaq.com/market-activity",
    "msn-finance": "https://www.msn.com/en-us/money",
    "msn-finance-live": "https://www.msn.com/en-us/money",
    "tencent-realtime": "https://gu.qq.com/",
    "tencent-minute": "https://gu.qq.com/",
    "nasdaq-public-chart": "https://www.nasdaq.com/market-activity",
    "yahoo-chart-1m": "https://finance.yahoo.com/",
}


@dataclass(frozen=True)
class DemoMarketConfig:
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    seed: int = 7
    tickers: tuple[str, ...] = DEFAULT_TICKERS


def _resize(values: list[float], length: int) -> np.ndarray:
    return np.resize(np.asarray(values, dtype=float), length)


def generate_demo_prices(config: DemoMarketConfig = DemoMarketConfig()) -> pd.DataFrame:
    """Generate reproducible stock-like prices for learning.

    The data is synthetic on purpose. It lets you run the project without API keys,
    vendor limits, adjusted-price quirks, or missing delisting data.
    """

    dates = pd.bdate_range(config.start, config.end)
    if len(dates) < 260:
        raise ValueError("Use at least one year of business-day data.")

    tickers = list(config.tickers)
    count = len(tickers)
    rng = np.random.default_rng(config.seed)
    n = len(dates)

    annual_drifts = _resize([0.11, 0.07, 0.09, 0.04, 0.14, 0.06], count)
    annual_vols = _resize([0.24, 0.16, 0.30, 0.11, 0.34, 0.19], count)
    betas = _resize([1.15, 0.75, 1.30, 0.45, 1.55, 0.90], count)

    cycle = np.sin(np.linspace(0, 6 * np.pi, n))
    regime_bonus = np.where(cycle >= 0, 0.03, -0.02)
    market_returns = rng.normal(
        loc=(0.06 + regime_bonus) / 252,
        scale=0.13 / np.sqrt(252),
        size=n,
    )

    returns = np.empty((n, count))
    for i in range(count):
        idiosyncratic = rng.normal(
            loc=annual_drifts[i] / 252,
            scale=annual_vols[i] / np.sqrt(252),
            size=n,
        )
        rare_down_days = rng.random(n) < 0.004
        jumps = np.where(rare_down_days, rng.normal(-0.04, 0.025, n), 0.0)
        returns[:, i] = betas[i] * market_returns + idiosyncratic + jumps

    returns = np.clip(returns, -0.18, 0.18)
    prices = 100.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers).round(2)


def generate_demo_ohlcv(
    symbol: str = "ALPHA",
    config: DemoMarketConfig = DemoMarketConfig(),
) -> pd.DataFrame:
    """Generate reproducible daily OHLCV data for one demo symbol."""

    symbol = symbol.strip().upper()
    close_table = generate_demo_prices(config)
    if symbol not in close_table.columns:
        raise ValueError(f"Unknown demo symbol: {symbol}")

    close = close_table[symbol]
    symbol_seed = sum(ord(char) for char in symbol)
    rng = np.random.default_rng(config.seed + symbol_seed)
    overnight_move = rng.normal(0.0, 0.004, len(close))
    open_price = close.shift(1).fillna(close.iloc[0]) * np.exp(overnight_move)
    intraday_range = np.abs(rng.normal(0.009, 0.004, len(close)))
    high = np.maximum(open_price, close) * (1.0 + intraday_range)
    low = np.minimum(open_price, close) * np.maximum(1.0 - intraday_range, 0.01)
    volume = rng.lognormal(mean=np.log(2_000_000), sigma=0.35, size=len(close))

    return clean_ohlcv_table(
        pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume.round(),
            },
            index=close.index,
        )
    )


def clean_price_table(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(how="all").ffill().dropna(how="any")

    if prices.empty:
        raise ValueError("No usable price rows after cleaning.")
    if prices.columns.empty:
        raise ValueError("No price columns found.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices


def clean_ohlcv_table(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize one-symbol daily bars to Date + OHLCV.

    Rows without complete OHLC values are removed. A missing volume is treated as
    zero because some CSV vendors omit it. Rolling-indicator NaNs are handled in
    the strategy module, not here.
    """

    if data.empty:
        raise ValueError("OHLCV data is empty.")

    frame = data.copy()
    column_lookup = {str(column).strip().lower(): column for column in frame.columns}
    date_column = column_lookup.get("date")
    if date_column is not None:
        frame = frame.set_index(date_column)
        column_lookup = {
            str(column).strip().lower(): column for column in frame.columns
        }

    missing = [column for column in OHLCV_COLUMNS if column.lower() not in column_lookup]
    if missing == ["Volume"]:
        frame["Volume"] = 0.0
        column_lookup["volume"] = "Volume"
        missing = []
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {', '.join(missing)}")

    normalized = pd.DataFrame(
        {
            column: pd.to_numeric(frame[column_lookup[column.lower()]], errors="coerce")
            for column in OHLCV_COLUMNS
        },
        index=frame.index,
    )
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized.loc[~normalized.index.isna()]
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = normalized.sort_index()
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized["Volume"] = normalized["Volume"].fillna(0.0)

    if normalized.empty:
        raise ValueError("No usable OHLCV rows after cleaning.")
    if (normalized[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive after adjustment.")
    if (normalized["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    normalized.index.name = "Date"
    return normalized


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load adjusted close prices from a CSV file.

    Expected shape:

    date,AAA,BBB,CCC
    2024-01-02,100.0,50.0,80.0
    2024-01-03,101.2,49.8,81.0
    """

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_price_table(prices)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load a Date, Open, High, Low, Close, Volume CSV file."""

    return clean_ohlcv_table(pd.read_csv(path))


def _import_optional(module_name: str, package_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{package_name}'. "
            f"Install it with: pip install {package_name}"
        ) from exc


def parse_symbols(symbols: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        raw_symbols = re.split(r"[\s,]+", symbols.strip())
    else:
        raw_symbols = list(symbols)

    parsed = tuple(symbol.strip() for symbol in raw_symbols if symbol.strip())
    if not parsed:
        raise ValueError("At least one symbol is required.")
    return parsed


def normalize_a_share_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    cleaned = cleaned.removeprefix("SH").removeprefix("SZ").removeprefix("BJ")
    cleaned = cleaned.split(".")[0]
    digits = "".join(char for char in cleaned if char.isdigit())
    if 1 <= len(digits) <= 6:
        digits = digits.zfill(6)
    if len(digits) != 6:
        raise ValueError(f"A-share symbol should be a 6-digit code: {symbol}")
    return digits


def provider_display_name(provider: object) -> str:
    key = str(provider or "unknown")
    return PROVIDER_LABELS.get(key, key)


def provider_source_details(provider: object, source_url: str = "") -> dict[str, str]:
    key = str(provider or "unknown")
    return {
        "id": key,
        "label": provider_display_name(key),
        "url": source_url or PROVIDER_URLS.get(key, ""),
    }


@contextmanager
def _available_console_streams():
    """Give console-oriented libraries safe streams inside a windowed EXE."""

    original_stdout, original_stderr = sys.stdout, sys.stderr
    if sys.stdout is None:
        sys.stdout = NullTextStream()
    if sys.stderr is None:
        sys.stderr = NullTextStream()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def summarize_error(error: Exception, limit: int = 180) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    message = re.sub(r"\s+with url:\s+\S+", "", message, flags=re.IGNORECASE)
    message = re.sub(r"\s+\(Caused by .*$", "", message, flags=re.IGNORECASE)
    if len(message) > limit:
        message = message[: limit - 3] + "..."
    return f"{error.__class__.__name__}: {message}"


def _provider_error(
    market: str,
    symbol: str,
    attempts: list[tuple[str, Exception]],
) -> RuntimeError:
    providers = "、".join(name for name, _error in attempts)
    details = "；".join(
        f"{name}（{summarize_error(error)}）" for name, error in attempts
    )
    return RuntimeError(
        f"无法获取{market} {symbol} 的历史行情。已尝试：{providers}。\n"
        f"失败摘要：{details}\n"
        "请检查网络或代理设置后重试，也可以切换数据源或导入 OHLCV CSV。"
    )


def _read_http_text(
    request: urlrequest.Request,
    *,
    timeout: float = 20,
    attempts: int = 2,
) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.35 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _a_share_market_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return "sh" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return "sz" + symbol


def normalize_hk_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().removesuffix(".HK")
    cleaned = cleaned.removeprefix("HK")
    digits = "".join(character for character in cleaned if character.isdigit())
    if not digits or len(digits) > 5:
        raise ValueError(f"港股代码应为 1 至 5 位数字：{symbol}")
    number = str(int(digits))
    return (number.zfill(4) if len(number) <= 4 else number) + ".HK"


def _hk_market_symbol(symbol: str) -> str:
    normalized = normalize_hk_symbol(symbol)
    digits = normalized.removesuffix(".HK")
    return "hk" + digits.zfill(5)


def _fetch_tencent_ohlcv_direct(
    symbol: str,
    start: str,
    end: str,
    adjust: str,
    *,
    market: str = "a-share",
) -> pd.DataFrame:
    """Fetch Tencent JSON directly without AkShare's console progress bar."""

    normalized_market = market.strip().lower()
    if normalized_market == "a-share":
        normalized_symbol = normalize_a_share_symbol(symbol)
        market_symbol = _a_share_market_symbol(normalized_symbol)
        provider = "tencent-direct"
    elif normalized_market == "hk":
        normalized_symbol = normalize_hk_symbol(symbol)
        market_symbol = _hk_market_symbol(normalized_symbol)
        provider = "tencent-hk-direct"
    else:
        raise ValueError("腾讯日线直连目前支持 A 股和港股。")
    start_timestamp = pd.Timestamp(start).normalize()
    end_timestamp = pd.Timestamp(end).normalize()
    if start_timestamp > end_timestamp:
        raise ValueError("行情开始日期不能晚于结束日期。")

    tx_adjust = "" if adjust == "none" else adjust
    expected_key = f"{tx_adjust}day" if tx_adjust else "day"
    frames: list[pd.DataFrame] = []
    endpoint_errors: list[tuple[str, Exception]] = []
    used_endpoint = ""
    used_row_keys: set[str] = set()

    for year in range(start_timestamp.year, end_timestamp.year + 1):
        year_loaded = False
        year_errors: list[tuple[str, Exception]] = []
        for endpoint in TENCENT_KLINE_ENDPOINTS:
            params = {
                "_var": f"kline_day{tx_adjust}{year}",
                "param": (
                    f"{market_symbol},day,{year}-01-01,{year}-12-31,"
                    f"320,{tx_adjust}"
                ),
                "r": "0.8205512681390605",
            }
            request = urlrequest.Request(
                f"{endpoint}?{urlparse.urlencode(params)}",
                headers={
                    "User-Agent": "Mozilla/5.0 TradingAgentsCN/1.0",
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": f"https://gu.qq.com/{market_symbol}/zs",
                },
            )
            try:
                body = _read_http_text(request, timeout=18, attempts=2)
                marker = body.find("={")
                if marker < 0:
                    raise ValueError("腾讯返回内容不是有效的 K 线 JSON。")
                payload = json.loads(body[marker + 1 :])
                stock_data = payload.get("data", {}).get(market_symbol, {})
                row_key = expected_key
                rows = stock_data.get(row_key, [])
                if not rows:
                    for fallback_key in ("day", "qfqday", "hfqday"):
                        if stock_data.get(fallback_key):
                            row_key = fallback_key
                            rows = stock_data[fallback_key]
                            break
                if rows:
                    frame = pd.DataFrame(
                        [row[:6] for row in rows],
                        columns=["date", "open", "close", "high", "low", "amount"],
                    )
                    frames.append(frame)
                    used_row_keys.add(row_key)
                used_endpoint = endpoint
                year_loaded = True
                break
            except Exception as exc:
                year_errors.append((endpoint, exc))
        if not year_loaded:
            endpoint_errors.extend(year_errors)
            break

    if endpoint_errors:
        summary = "; ".join(
            f"{urlparse.urlparse(endpoint).netloc}: {summarize_error(error)}"
            for endpoint, error in endpoint_errors
        )
        raise RuntimeError("腾讯直连失败：" + summary)
    if not frames:
        raise ValueError("腾讯直连未返回指定日期范围内的行情。")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.loc[
        raw["date"].between(start_timestamp, end_timestamp, inclusive="both")
    ]
    if raw.empty:
        raise ValueError("腾讯直连未返回指定日期范围内的行情。")
    bars = clean_ohlcv_table(
        raw.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "amount": "Volume",
            }
        )[["Date", *OHLCV_COLUMNS]]
    )
    bars.attrs["provider"] = provider
    bars.attrs["provider_endpoint"] = urlparse.urlparse(used_endpoint).netloc
    bars.attrs["market"] = normalized_market
    bars.attrs["symbol"] = normalized_symbol
    bars.attrs["adjustment_key"] = ",".join(sorted(used_row_keys))
    if expected_key not in used_row_keys:
        bars.attrs["adjustment_fallback"] = True
    return bars


def fetch_a_share_ohlcv(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch one A-share symbol as a normalized daily OHLCV table."""

    normalized_symbol = normalize_a_share_symbol(symbol)
    start_date = pd.Timestamp(start).strftime("%Y%m%d")
    end_date = pd.Timestamp(end).strftime("%Y%m%d")
    ak_adjust = "" if adjust == "none" else adjust
    attempts: list[tuple[str, Exception]] = []

    try:
        akshare = _import_optional("akshare", "akshare")
    except Exception as exc:
        akshare = None
        attempts.append(("AkShare", exc))

    raw = pd.DataFrame()
    if akshare is not None:
        try:
            raw = akshare.stock_zh_a_hist(
                symbol=normalized_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=ak_adjust,
                timeout=15,
            )
        except Exception as exc:
            attempts.append(("东方财富", exc))

    required = {"日期", "开盘", "最高", "最低", "收盘", "成交量"}
    if not raw.empty and required.issubset(raw.columns):
        bars = clean_ohlcv_table(
            raw.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "最高": "High",
                    "最低": "Low",
                    "收盘": "Close",
                    "成交量": "Volume",
                }
            )[["Date", *OHLCV_COLUMNS]]
        )
        bars.attrs["provider"] = "akshare-eastmoney"
        return bars

    if akshare is not None and not any(name == "东方财富" for name, _ in attempts):
        attempts.append(
            (
                "东方财富",
                ValueError(
                    "接口未返回可用行或缺少字段：" + ", ".join(map(str, raw.columns))
                ),
            )
        )

    try:
        return _fetch_tencent_ohlcv_direct(
            normalized_symbol,
            start_date,
            end_date,
            adjust,
        )
    except Exception as exc:
        attempts.append(("腾讯直连", exc))

    if akshare is not None:
        market_symbol = _a_share_market_symbol(normalized_symbol)
        try:
            with _available_console_streams():
                fallback = akshare.stock_zh_a_hist_tx(
                    symbol=market_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=ak_adjust,
                    timeout=15,
                )
            fallback_required = {"date", "open", "high", "low", "close", "amount"}
            if fallback.empty or not fallback_required.issubset(fallback.columns):
                raise ValueError(
                    "接口未返回可用行或缺少字段："
                    + ", ".join(map(str, fallback.columns))
                )
            bars = clean_ohlcv_table(
                fallback.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "amount": "Volume",
                    }
                )[["Date", *OHLCV_COLUMNS]]
            )
            bars.attrs["provider"] = "akshare-tencent"
            return bars
        except Exception as exc:
            attempts.append(("腾讯 AkShare", exc))

    raise _provider_error("A 股", normalized_symbol, attempts)


def fetch_hk_ohlcv(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch one Hong Kong listed stock or ETF through independent providers."""

    normalized_symbol = normalize_hk_symbol(symbol)
    attempts: list[tuple[str, Exception]] = []
    try:
        return _fetch_tencent_ohlcv_direct(
            normalized_symbol,
            start,
            end,
            adjust,
            market="hk",
        )
    except Exception as exc:
        attempts.append(("腾讯证券", exc))

    try:
        return fetch_msn_ohlcv(
            normalized_symbol,
            "hk",
            start,
            end,
        )
    except Exception as exc:
        attempts.append(("Microsoft Finance", exc))

    try:
        akshare = _import_optional("akshare", "akshare")
        digits = normalized_symbol.removesuffix(".HK").zfill(5)
        raw = akshare.stock_hk_hist(
            symbol=digits,
            period="daily",
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            adjust="" if adjust == "none" else adjust,
        )
        required = {"日期", "开盘", "最高", "最低", "收盘", "成交量"}
        if raw is None or raw.empty or not required.issubset(raw.columns):
            raise ValueError("东方财富港股接口未返回完整 OHLCV。")
        bars = clean_ohlcv_table(
            raw.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "最高": "High",
                    "最低": "Low",
                    "收盘": "Close",
                    "成交量": "Volume",
                }
            )[["Date", *OHLCV_COLUMNS]]
        )
        bars.attrs.update(
            provider="akshare-eastmoney",
            market="hk",
            symbol=normalized_symbol,
        )
        return bars
    except Exception as exc:
        attempts.append(("东方财富 / AkShare", exc))
    raise _provider_error("港股", normalized_symbol, attempts)


def _fetch_stooq_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a US ticker from Stooq's public historical CSV endpoint."""

    if not re.fullmatch(r"[A-Z0-9-]+", symbol):
        raise ValueError("Stooq fallback supports simple US ticker symbols only.")
    stooq_symbol = f"{symbol.lower()}.us"
    query = urlparse.urlencode(
        {
            "s": stooq_symbol,
            "d1": pd.Timestamp(start).strftime("%Y%m%d"),
            "d2": pd.Timestamp(end).strftime("%Y%m%d"),
            "i": "d",
        }
    )
    http_request = urlrequest.Request(
        f"https://stooq.com/q/d/l/?{query}",
        headers={"User-Agent": "Mozilla/5.0 TradingAgentsCN/1.0"},
    )
    body = _read_http_text(http_request, timeout=20, attempts=2)
    if "<html" in body[:200].lower():
        raise RuntimeError("Stooq returned a browser verification page instead of CSV.")
    raw = pd.read_csv(io.StringIO(body))
    bars = clean_ohlcv_table(raw)
    bars.attrs["provider"] = "stooq"
    return bars


def _fetch_nasdaq_api_ohlcv(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch US daily bars from Nasdaq's public market-activity API."""

    if not re.fullmatch(r"[A-Z0-9.^=-]+", symbol):
        raise ValueError("Nasdaq API received an invalid ticker symbol.")
    api_symbol = NASDAQ_INDEX_SYMBOLS.get(symbol, symbol.replace("-", "."))
    if symbol in NASDAQ_INDEX_SYMBOLS:
        asset_classes = ("index", "stocks", "etf")
    elif symbol in NASDAQ_ETF_SYMBOLS:
        asset_classes = ("etf", "stocks", "index")
    else:
        asset_classes = ("stocks", "etf", "index")

    start_date = pd.Timestamp(start).date().isoformat()
    end_date = pd.Timestamp(end).date().isoformat()
    attempts: list[tuple[str, Exception]] = []
    for asset_class in asset_classes:
        query = urlparse.urlencode(
            {
                "assetclass": asset_class,
                "fromdate": start_date,
                "todate": end_date,
                "limit": 5000,
            }
        )
        endpoint = (
            f"{NASDAQ_API_BASE}/{urlparse.quote(api_symbol, safe='.')}/historical?{query}"
        )
        request = urlrequest.Request(
            endpoint,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": (
                    "https://www.nasdaq.com/market-activity/"
                    f"{asset_class}/{api_symbol.lower()}/historical"
                ),
            },
        )
        try:
            payload = json.loads(
                _read_http_text(request, timeout=20, attempts=2)
            )
            status = payload.get("status") or {}
            if status.get("rCode") not in {None, 200}:
                messages = status.get("bCodeMessage") or []
                message = "; ".join(
                    str(item.get("errorMessage", item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in messages
                )
                raise RuntimeError(message or "Nasdaq API returned an error status.")
            data = payload.get("data") or {}
            rows = (data.get("tradesTable") or {}).get("rows") or []
            if not rows:
                raise ValueError("Nasdaq API returned no historical rows.")
            raw = pd.DataFrame(rows).rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            required = {"Date", *OHLCV_COLUMNS}
            if not required.issubset(raw.columns):
                raise ValueError(
                    "Nasdaq API response is missing OHLCV fields: "
                    + ", ".join(map(str, raw.columns))
                )
            for column in OHLCV_COLUMNS:
                raw[column] = pd.to_numeric(
                    raw[column]
                    .astype(str)
                    .str.replace(r"[$,]", "", regex=True)
                    .replace({"N/A": np.nan, "--": np.nan}),
                    errors="coerce",
                )
            bars = clean_ohlcv_table(raw[["Date", *OHLCV_COLUMNS]])
            bars.attrs.update(
                provider="nasdaq-api",
                provider_asset_class=asset_class,
                provider_endpoint="api.nasdaq.com",
            )
            return bars
        except Exception as exc:
            attempts.append((asset_class, exc))

    details = "; ".join(
        f"{asset_class}: {summarize_error(error)}"
        for asset_class, error in attempts
    )
    raise RuntimeError("Nasdaq 官方历史行情不可用：" + details)


def _fetch_yahoo_chart_ohlcv(
    symbol: str,
    start: str,
    end: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Fetch Yahoo's JSON chart endpoint without yfinance session state."""

    if not re.fullmatch(r"[A-Z0-9.^=-]+", symbol):
        raise ValueError("Yahoo chart fallback received an invalid ticker symbol.")
    start_epoch = int(
        datetime.combine(pd.Timestamp(start).date(), datetime.min.time(), timezone.utc).timestamp()
    )
    end_epoch = int(
        datetime.combine(
            pd.Timestamp(end).date() + timedelta(days=1),
            datetime.min.time(),
            timezone.utc,
        ).timestamp()
    )
    query = urlparse.urlencode(
        {
            "period1": start_epoch,
            "period2": end_epoch,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    http_request = urlrequest.Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 TradingAgentsCN/1.0",
            "Accept": "application/json",
        },
    )
    payload = json.loads(_read_http_text(http_request, timeout=20, attempts=2))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo chart returned no results.")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quote_rows:
        raise ValueError("Yahoo chart returned no daily quote rows.")

    quote = quote_rows[0]
    timezone_name = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(timezone_name)
        .tz_localize(None)
        .normalize()
    )
    raw = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=dates,
    )
    if auto_adjust:
        adjusted_rows = result.get("indicators", {}).get("adjclose") or []
        if adjusted_rows:
            adjusted_close = pd.Series(
                adjusted_rows[0].get("adjclose", []), index=dates, dtype=float
            )
            raw_close = pd.to_numeric(raw["Close"], errors="coerce")
            adjustment_factor = adjusted_close.div(raw_close).replace(
                [np.inf, -np.inf], np.nan
            )
            for column in ("Open", "High", "Low", "Close"):
                raw[column] = pd.to_numeric(raw[column], errors="coerce") * adjustment_factor

    bars = clean_ohlcv_table(raw)
    bars.attrs["provider"] = "yahoo-chart"
    return bars


def fetch_a_share_prices(
    symbols: str | list[str] | tuple[str, ...],
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch A-share daily adjusted close prices with AkShare.

    symbols examples: 600519, 000001, 300750.
    adjust: qfq = front-adjusted, hfq = back-adjusted, none = unadjusted.
    """

    parsed_symbols = tuple(normalize_a_share_symbol(symbol) for symbol in parse_symbols(symbols))

    series_by_symbol: dict[str, pd.Series] = {}
    providers: dict[str, str] = {}
    for symbol in parsed_symbols:
        bars = fetch_a_share_ohlcv(symbol, start, end, adjust)
        series_by_symbol[symbol] = bars["Close"].rename(symbol)
        providers[symbol] = str(bars.attrs.get("provider", "unknown"))

    prices = clean_price_table(pd.DataFrame(series_by_symbol))
    prices.attrs["providers"] = providers
    return prices


def _extract_yfinance_close(raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No Nasdaq data returned.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))

        if "Close" in level0:
            close = raw["Close"].copy()
            close = close[[symbol for symbol in symbols if symbol in close.columns]]
        elif "Close" in level1:
            close = pd.DataFrame(
                {
                    symbol: raw[symbol]["Close"]
                    for symbol in symbols
                    if symbol in raw.columns.get_level_values(0)
                    and "Close" in raw[symbol].columns
                }
            )
        else:
            raise ValueError(f"Cannot find Close columns in yfinance data: {raw.columns}")
    else:
        if "Close" not in raw.columns:
            raise ValueError(f"Cannot find Close column in yfinance data: {raw.columns}")
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})

    if close.empty:
        raise ValueError("No usable Close prices in Nasdaq data.")
    return close


def _extract_yfinance_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(f"No Nasdaq/US data returned for {symbol}.")

    if not isinstance(raw.columns, pd.MultiIndex):
        return clean_ohlcv_table(raw)

    level0 = set(raw.columns.get_level_values(0))
    level1 = set(raw.columns.get_level_values(1))
    if set(OHLCV_COLUMNS).issubset(level0):
        extracted = pd.DataFrame(
            {
                column: raw[column][symbol]
                if isinstance(raw[column], pd.DataFrame) and symbol in raw[column].columns
                else raw[column].squeeze()
                for column in OHLCV_COLUMNS
            },
            index=raw.index,
        )
    elif symbol in level0 and set(OHLCV_COLUMNS).issubset(level1):
        extracted = raw[symbol][list(OHLCV_COLUMNS)].copy()
    else:
        raise ValueError(f"Cannot find OHLCV columns in yfinance data: {raw.columns}")
    return clean_ohlcv_table(extracted)


def fetch_nasdaq_ohlcv(
    symbol: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch one Nasdaq/US-listed symbol with four independent fallbacks."""

    normalized_symbol = symbol.strip().upper()
    attempts: list[tuple[str, Exception]] = []
    try:
        return _fetch_nasdaq_api_ohlcv(normalized_symbol, start, end)
    except Exception as exc:
        attempts.append(("Nasdaq 官方", exc))

    try:
        return _fetch_yahoo_chart_ohlcv(
            normalized_symbol, start, end, auto_adjust
        )
    except Exception as exc:
        attempts.append(("Yahoo Chart", exc))

    try:
        return fetch_msn_ohlcv(
            normalized_symbol,
            "nasdaq",
            start,
            end,
        )
    except Exception as exc:
        attempts.append(("Microsoft Finance", exc))

    end_exclusive = (pd.Timestamp(end).date() + timedelta(days=1)).isoformat()
    try:
        yfinance = _import_optional("yfinance", "yfinance")
        raw = yfinance.download(
            tickers=normalized_symbol,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=end_exclusive,
            interval="1d",
            auto_adjust=auto_adjust,
            group_by="column",
            progress=False,
            threads=False,
            timeout=20,
        )
        bars = _extract_yfinance_ohlcv(raw, normalized_symbol)
        bars.attrs["provider"] = "yfinance"
        return bars
    except Exception as exc:
        attempts.append(("yfinance", exc))

    try:
        return _fetch_stooq_ohlcv(normalized_symbol, start, end)
    except Exception as exc:
        attempts.append(("Stooq", exc))
    raise _provider_error("美股/全球市场", normalized_symbol, attempts)


def fetch_global_ohlcv(
    symbol: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch a non-US global stock, ETF, fund or index with resilient fallbacks."""

    normalized_symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,24}", normalized_symbol):
        raise ValueError("全球市场代码格式无效。")
    attempts: list[tuple[str, Exception]] = []
    try:
        return fetch_msn_ohlcv(
            normalized_symbol,
            "global",
            start,
            end,
        )
    except Exception as exc:
        attempts.append(("Microsoft Finance", exc))

    try:
        return _fetch_yahoo_chart_ohlcv(
            normalized_symbol, start, end, auto_adjust
        )
    except Exception as exc:
        attempts.append(("Yahoo Chart", exc))

    end_exclusive = (pd.Timestamp(end).date() + timedelta(days=1)).isoformat()
    try:
        yfinance = _import_optional("yfinance", "yfinance")
        raw = yfinance.download(
            tickers=normalized_symbol,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=end_exclusive,
            interval="1d",
            auto_adjust=auto_adjust,
            group_by="column",
            progress=False,
            threads=False,
            timeout=20,
        )
        bars = _extract_yfinance_ohlcv(raw, normalized_symbol)
        bars.attrs["provider"] = "yfinance"
        return bars
    except Exception as exc:
        attempts.append(("yfinance", exc))
    raise _provider_error("全球市场", normalized_symbol, attempts)


def fetch_market_ohlcv(
    market: str,
    symbol: str,
    start: str,
    end: str,
    *,
    adjust: str = "qfq",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    normalized_market = market.strip().lower()
    if normalized_market == "a-share":
        return fetch_a_share_ohlcv(symbol, start, end, adjust)
    if normalized_market in {"nasdaq", "us", "usa"}:
        return fetch_nasdaq_ohlcv(symbol, start, end, auto_adjust)
    if normalized_market in {"hk", "hong-kong"}:
        return fetch_hk_ohlcv(symbol, start, end, adjust)
    if normalized_market in {"global", "world", "international"}:
        return fetch_global_ohlcv(symbol, start, end, auto_adjust)
    raise ValueError("市场必须是 a-share、nasdaq、hk 或 global。")


def fetch_nasdaq_prices(
    symbols: str | list[str] | tuple[str, ...],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch Nasdaq/US-listed daily close prices through resilient providers."""

    parsed_symbols = tuple(symbol.upper() for symbol in parse_symbols(symbols))
    series_by_symbol: dict[str, pd.Series] = {}
    providers: dict[str, str] = {}
    for symbol in parsed_symbols:
        bars = fetch_nasdaq_ohlcv(symbol, start, end, auto_adjust)
        series_by_symbol[symbol] = bars["Close"].rename(symbol)
        providers[symbol] = str(bars.attrs.get("provider", "unknown"))
    prices = clean_price_table(pd.DataFrame(series_by_symbol))
    prices.attrs["providers"] = providers
    return prices
