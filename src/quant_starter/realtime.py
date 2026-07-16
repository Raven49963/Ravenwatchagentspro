from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import re
import time
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import (
    NASDAQ_API_BASE,
    NASDAQ_ETF_SYMBOLS,
    NASDAQ_INDEX_SYMBOLS,
    _a_share_market_symbol,
    _hk_market_symbol,
    normalize_hk_symbol,
    normalize_a_share_symbol,
    summarize_error,
)
from .global_market import (
    ASSET_TYPE_LABELS,
    MSNInstrument,
    fetch_msn_intraday,
    quote_timestamp,
    resolve_msn_instrument,
)
from .symbols import preset_for_symbol


TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q={symbol}"
TENCENT_MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
YAHOO_INTRADAY_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class MarketSnapshot:
    market: str
    symbol: str
    name: str
    currency: str
    price: float
    previous_close: float
    change: float
    change_pct: float
    open: float
    high: float
    low: float
    volume: float
    amount: float
    timestamp: str
    session_status: str
    provider: str
    delayed_seconds: int | None = None
    exchange: str = ""
    country: str = ""
    asset_type: str = "stock"
    asset_type_label: str = "股票"
    timezone: str = ""
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealtimeMarketData:
    snapshot: MarketSnapshot
    intraday: pd.DataFrame


def _read_http_bytes(
    request: urlrequest.Request,
    *,
    timeout: float = 15,
    attempts: int = 2,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _field(fields: list[str], position: int, default: float = 0.0) -> float:
    return _finite(fields[position] if position < len(fields) else None, default)


def _session_state(
    timestamp: pd.Timestamp,
    market: str,
    timezone_name: str = "",
) -> tuple[str, int | None]:
    if timestamp.tzinfo is None:
        fallback_timezone = (
            "Asia/Shanghai"
            if market == "a-share"
            else "Asia/Hong_Kong"
            if market == "hk"
            else "America/New_York"
        )
        timestamp = timestamp.tz_localize(timezone_name or fallback_timezone)
    now = pd.Timestamp.now(tz=timestamp.tz)
    delay_seconds = max(0, int((now - timestamp).total_seconds()))
    same_day = now.date() == timestamp.date()
    minute = now.hour * 60 + now.minute
    if market == "a-share":
        in_session = 570 <= minute <= 690 or 780 <= minute <= 900
    elif market == "hk":
        in_session = 570 <= minute <= 720 or 780 <= minute <= 960
    else:
        in_session = 540 <= minute <= 1_020
    if same_day and in_session and delay_seconds <= 600:
        return "open", delay_seconds
    if same_day and in_session:
        return "delayed", delay_seconds
    return "closed", None


def _tencent_context(symbol: str, market: str) -> tuple[str, str, str, str, float]:
    if market == "a-share":
        normalized = normalize_a_share_symbol(symbol)
        return normalized, _a_share_market_symbol(normalized), "CNY", "Asia/Shanghai", 100.0
    if market == "hk":
        normalized = normalize_hk_symbol(symbol)
        return normalized, _hk_market_symbol(normalized), "HKD", "Asia/Hong_Kong", 1.0
    raise ValueError("腾讯实时行情目前支持 A 股和港股。")


def _tencent_quote(symbol: str, market: str = "a-share") -> MarketSnapshot:
    normalized, market_symbol, currency, timezone_name, volume_multiplier = (
        _tencent_context(symbol, market)
    )
    request = urlrequest.Request(
        TENCENT_QUOTE_URL.format(symbol=market_symbol),
        headers={
            "User-Agent": "Mozilla/5.0 RavenWatchAgentsCN/1.6",
            "Accept": "text/plain,*/*",
            "Referer": f"https://gu.qq.com/{market_symbol}/gp",
        },
    )
    body = _read_http_bytes(request, timeout=12, attempts=2).decode(
        "gb18030", errors="replace"
    )
    match = re.search(r'="(.*)";?\s*$', body.strip())
    if match is None:
        raise ValueError("腾讯实时行情返回格式无效。")
    fields = match.group(1).split("~")
    if len(fields) < 35:
        raise ValueError("腾讯实时行情字段不完整。")

    price = _field(fields, 3)
    previous_close = _field(fields, 4)
    if price <= 0 or previous_close <= 0:
        raise ValueError("腾讯实时行情没有有效价格。")
    timestamp_text = fields[30] if len(fields) > 30 else ""
    timestamp = None
    for pattern in ("%Y%m%d%H%M%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            timestamp = pd.Timestamp(
                datetime.strptime(timestamp_text, pattern),
                tz=ZoneInfo(timezone_name),
            )
            break
        except ValueError:
            continue
    if timestamp is None:
        timestamp = pd.Timestamp.now(tz=timezone_name)

    volume_lots = _field(fields, 6)
    amount = 0.0
    if len(fields) > 35:
        deal_parts = fields[35].split("/")
        if len(deal_parts) >= 3:
            amount = _finite(deal_parts[2])
    if amount <= 0:
        amount = _field(fields, 37) * (10_000 if market == "a-share" else 1)
    change = price - previous_close
    session_status, delayed_seconds = _session_state(
        timestamp, market, timezone_name
    )
    detected_asset_type = (
        "etf"
        if "基金" in (fields[1] if len(fields) > 1 else "")
        or any("ETF" in field.upper() for field in fields)
        else "stock"
    )
    preset = preset_for_symbol(market, normalized)
    asset_type = preset.asset_type if preset is not None else detected_asset_type
    return MarketSnapshot(
        market=market,
        symbol=normalized,
        name=fields[1] or normalized,
        currency=currency,
        price=round(price, 4),
        previous_close=round(previous_close, 4),
        change=round(change, 4),
        change_pct=round(change / previous_close, 6),
        open=round(_field(fields, 5, previous_close), 4),
        high=round(_field(fields, 33, price), 4),
        low=round(_field(fields, 34, price), 4),
        volume=round(volume_lots * volume_multiplier, 0),
        amount=round(amount, 2),
        timestamp=timestamp.isoformat(),
        session_status=session_status,
        provider="tencent-realtime",
        delayed_seconds=delayed_seconds,
        exchange=(
            preset.exchange
            if preset is not None and preset.exchange
            else "沪深交易所" if market == "a-share" else "香港交易所"
        ),
        country=(
            preset.country
            if preset is not None and preset.country
            else "CN" if market == "a-share" else "HK"
        ),
        asset_type=asset_type,
        asset_type_label=ASSET_TYPE_LABELS.get(asset_type, "证券"),
        timezone=timezone_name,
        source_url=f"https://gu.qq.com/{market_symbol}/gp",
    )


def _tencent_intraday(symbol: str, market: str = "a-share") -> pd.DataFrame:
    normalized, market_symbol, _currency, timezone_name, volume_multiplier = (
        _tencent_context(symbol, market)
    )
    query = urlparse.urlencode({"code": market_symbol})
    request = urlrequest.Request(
        f"{TENCENT_MINUTE_URL}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 RavenWatchAgentsCN/1.6",
            "Accept": "application/json,text/plain,*/*",
            "Referer": f"https://gu.qq.com/{market_symbol}/gp",
        },
    )
    payload = json.loads(
        _read_http_bytes(request, timeout=12, attempts=2).decode(
            "utf-8", errors="replace"
        )
    )
    if payload.get("code") not in {0, "0"}:
        raise RuntimeError(f"腾讯分时接口错误：{payload.get('msg', 'unknown')}")
    stock = payload.get("data", {}).get(market_symbol, {})
    minute_data = stock.get("data", {})
    date_text = str(minute_data.get("date", ""))
    rows = minute_data.get("data") or []
    if not rows or not re.fullmatch(r"\d{8}", date_text):
        raise ValueError("腾讯分时接口未返回有效分钟数据。")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parts = str(row).split()
        if len(parts) < 4 or not re.fullmatch(r"\d{4}", parts[0]):
            continue
        timestamp = pd.Timestamp(
            datetime.strptime(date_text + parts[0], "%Y%m%d%H%M"),
            tz=ZoneInfo(timezone_name),
        )
        parsed.append(
            {
                "DateTime": timestamp,
                "Price": _finite(parts[1]),
                "CumulativeVolume": _finite(parts[2]) * volume_multiplier,
                "CumulativeAmount": _finite(parts[3]),
            }
        )
    frame = pd.DataFrame(parsed).set_index("DateTime")
    frame = frame.loc[frame["Price"] > 0].copy()
    if frame.empty:
        raise ValueError("腾讯分时接口没有可用价格。")
    frame["Volume"] = frame["CumulativeVolume"].diff().fillna(
        frame["CumulativeVolume"]
    ).clip(lower=0)
    frame["Amount"] = frame["CumulativeAmount"].diff().fillna(
        frame["CumulativeAmount"]
    ).clip(lower=0)
    frame["AveragePrice"] = frame["CumulativeAmount"].div(
        frame["CumulativeVolume"].replace(0, np.nan)
    ).ffill()
    frame = frame[
        [
            "Price",
            "AveragePrice",
            "Volume",
            "CumulativeVolume",
            "Amount",
        ]
    ]
    frame.attrs.update(
        provider="tencent-minute",
        market=market,
        symbol=normalized,
        trading_date=date_text,
    )
    return frame


def _yahoo_intraday_result(symbol: str) -> dict[str, Any]:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", normalized):
        raise ValueError("美股代码格式无效。")
    query = urlparse.urlencode(
        {
            "interval": "1m",
            "range": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    request = urlrequest.Request(
        f"{YAHOO_INTRADAY_URL.format(symbol=normalized)}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 RavenWatchAgentsCN/1.6",
            "Accept": "application/json",
        },
    )
    payload = json.loads(
        _read_http_bytes(request, timeout=15, attempts=2).decode(
            "utf-8", errors="replace"
        )
    )
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo 分时接口错误：{chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo 分时接口没有返回数据。")
    return results[0]


def _yahoo_intraday_frame(result: dict[str, Any]) -> pd.DataFrame:
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quotes:
        raise ValueError("Yahoo 分时接口没有分钟报价。")
    quote = quotes[0]
    length = len(timestamps)
    timezone_name = result.get("meta", {}).get(
        "exchangeTimezoneName", "America/New_York"
    )

    def values(name: str) -> list[Any]:
        raw = list(quote.get(name) or [])
        return (raw + [None] * length)[:length]

    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone_name)
    frame = pd.DataFrame(
        {
            "Price": values("close"),
            "Open": values("open"),
            "High": values("high"),
            "Low": values("low"),
            "Volume": values("volume"),
        },
        index=index,
    )
    frame.index.name = "DateTime"
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(subset=["Price"])
    if frame.empty:
        raise ValueError("Yahoo 分时接口没有可用价格。")
    frame["Volume"] = frame["Volume"].fillna(0).clip(lower=0)
    frame["CumulativeVolume"] = frame["Volume"].cumsum()
    traded_value = frame["Price"] * frame["Volume"]
    frame["Amount"] = traded_value
    frame["AveragePrice"] = traded_value.cumsum().div(
        frame["CumulativeVolume"].replace(0, np.nan)
    ).fillna(frame["Price"].expanding().mean())
    meta = result.get("meta", {})
    frame.attrs.update(
        provider="yahoo-chart-1m",
        market="nasdaq",
        symbol=str(meta.get("symbol", "")),
        trading_date=frame.index[-1].strftime("%Y%m%d"),
    )
    return frame[
        [
            "Price",
            "AveragePrice",
            "Volume",
            "CumulativeVolume",
            "Amount",
            "Open",
            "High",
            "Low",
        ]
    ]


def _yahoo_snapshot(result: dict[str, Any], frame: pd.DataFrame) -> MarketSnapshot:
    meta = result.get("meta", {})
    symbol = str(meta.get("symbol", "")).upper()
    price = _finite(meta.get("regularMarketPrice"), _latest_price(frame))
    previous_close = _finite(
        meta.get("previousClose"),
        _finite(meta.get("chartPreviousClose"), price),
    )
    if price <= 0 or previous_close <= 0:
        raise ValueError("Yahoo 分时接口没有有效市场价格。")
    timezone_name = meta.get("exchangeTimezoneName", "America/New_York")
    market_time = meta.get("regularMarketTime")
    timestamp = (
        pd.Timestamp(int(market_time), unit="s", tz="UTC").tz_convert(timezone_name)
        if market_time
        else pd.Timestamp(frame.index[-1])
    )
    session_status, delayed_seconds = _session_state(timestamp, "nasdaq")
    change = price - previous_close
    opening = _finite(frame["Open"].dropna().iloc[0] if frame["Open"].notna().any() else None, previous_close)
    high = _finite(meta.get("regularMarketDayHigh"), _finite(frame["High"].max(), price))
    low = _finite(meta.get("regularMarketDayLow"), _finite(frame["Low"].min(), price))
    volume = _finite(meta.get("regularMarketVolume"), _finite(frame["Volume"].sum()))
    return MarketSnapshot(
        market="nasdaq",
        symbol=symbol,
        name=str(meta.get("longName") or meta.get("shortName") or symbol),
        currency=str(meta.get("currency") or "USD"),
        price=round(price, 4),
        previous_close=round(previous_close, 4),
        change=round(change, 4),
        change_pct=round(change / previous_close, 6),
        open=round(opening, 4),
        high=round(high, 4),
        low=round(low, 4),
        volume=round(volume, 0),
        amount=round(float(frame["Amount"].sum()), 2),
        timestamp=timestamp.isoformat(),
        session_status=session_status,
        provider="yahoo-chart-1m",
        delayed_seconds=delayed_seconds,
    )


def _latest_price(frame: pd.DataFrame) -> float:
    return _finite(frame["Price"].dropna().iloc[-1] if not frame.empty else None)


def _market_number(value: Any, default: float = 0.0) -> float:
    cleaned = re.sub(r"[$,%+,]", "", str(value or "")).strip()
    return _finite(cleaned, default)


def _nasdaq_asset_candidates(symbol: str) -> tuple[str, tuple[str, ...]]:
    api_symbol = NASDAQ_INDEX_SYMBOLS.get(symbol, symbol.replace("-", "."))
    if symbol in NASDAQ_INDEX_SYMBOLS:
        return api_symbol, ("index", "stocks", "etf")
    if symbol in NASDAQ_ETF_SYMBOLS:
        return api_symbol, ("etf", "stocks", "index")
    return api_symbol, ("stocks", "etf", "index")


def _nasdaq_intraday_result(symbol: str) -> tuple[dict[str, Any], str]:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]+", normalized):
        raise ValueError("Nasdaq 分时接口收到无效股票代码。")
    api_symbol, asset_classes = _nasdaq_asset_candidates(normalized)
    errors: list[tuple[str, Exception]] = []
    for asset_class in asset_classes:
        query = urlparse.urlencode({"assetclass": asset_class})
        endpoint = (
            f"{NASDAQ_API_BASE}/{urlparse.quote(api_symbol, safe='.')}/chart?{query}"
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
                    f"{asset_class}/{api_symbol.lower()}/real-time"
                ),
            },
        )
        try:
            payload = json.loads(
                _read_http_bytes(request, timeout=15, attempts=2).decode(
                    "utf-8", errors="replace"
                )
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
                raise RuntimeError(message or "Nasdaq 分时接口返回错误状态。")
            data = payload.get("data") or {}
            if not data.get("chart"):
                raise ValueError("Nasdaq 分时接口没有价格序列。")
            return data, asset_class
        except Exception as exc:
            errors.append((asset_class, exc))
    details = "; ".join(
        f"{asset}: {summarize_error(error)}" for asset, error in errors
    )
    raise RuntimeError("Nasdaq 分时行情不可用：" + details)


def _nasdaq_intraday_frame(data: dict[str, Any], asset_class: str) -> pd.DataFrame:
    rows = data.get("chart") or []
    raw = pd.DataFrame(
        {
            "timestamp": [item.get("x") for item in rows],
            "Price": [item.get("y") for item in rows],
        }
    )
    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"], unit="ms", utc=True, errors="coerce"
    )
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "Price"])
    raw = raw.loc[raw["Price"] > 0]
    if raw.empty:
        raise ValueError("Nasdaq 分时接口没有可用价格。")
    index = pd.DatetimeIndex(raw["timestamp"]).tz_convert("America/New_York")
    frame = pd.DataFrame({"Price": raw["Price"].to_numpy()}, index=index)
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = "DateTime"
    frame["AveragePrice"] = frame["Price"]
    frame["Volume"] = 0.0
    frame["CumulativeVolume"] = 0.0
    frame["Amount"] = 0.0
    frame["Open"] = frame["Price"]
    frame["High"] = frame["Price"]
    frame["Low"] = frame["Price"]
    frame.attrs.update(
        provider="nasdaq-public-chart",
        market="nasdaq",
        symbol=str(data.get("symbol", "")),
        asset_class=asset_class,
        volume_granularity="snapshot-only",
    )
    return frame


def _nasdaq_snapshot(
    data: dict[str, Any],
    frame: pd.DataFrame,
) -> MarketSnapshot:
    symbol = str(data.get("symbol", "")).upper()
    price = _market_number(data.get("lastSalePrice"), _latest_price(frame))
    previous_close = _market_number(data.get("previousClose"), price)
    if price <= 0 or previous_close <= 0:
        raise ValueError("Nasdaq 分时接口没有有效市场价格。")
    regular = frame.between_time("09:30", "16:00")
    trading = regular if not regular.empty else frame
    opening = _finite(trading["Price"].iloc[0], previous_close)
    high = _finite(trading["Price"].max(), price)
    low = _finite(trading["Price"].min(), price)
    volume = _market_number(data.get("volume"))
    timestamp = pd.Timestamp(frame.index[-1])
    session_status, delayed_seconds = _session_state(timestamp, "nasdaq")
    change = _market_number(data.get("netChange"), price - previous_close)
    return MarketSnapshot(
        market="nasdaq",
        symbol=symbol,
        name=str(data.get("company") or symbol),
        currency="USD",
        price=round(price, 4),
        previous_close=round(previous_close, 4),
        change=round(change, 4),
        change_pct=round(change / previous_close, 6),
        open=round(opening, 4),
        high=round(high, 4),
        low=round(low, 4),
        volume=round(volume, 0),
        amount=round(price * volume, 2),
        timestamp=timestamp.isoformat(),
        session_status=session_status,
        provider="nasdaq-public-chart",
        delayed_seconds=delayed_seconds,
        exchange="Nasdaq / US Market",
        country="US",
        asset_type=str(frame.attrs.get("asset_class", "stock")).removesuffix("s"),
        asset_type_label=ASSET_TYPE_LABELS.get(
            str(frame.attrs.get("asset_class", "stock")).removesuffix("s"),
            "证券",
        ),
        timezone="America/New_York",
        source_url="https://www.nasdaq.com/market-activity",
    )


def _msn_snapshot(
    instrument: MSNInstrument,
    quote: dict[str, Any],
    frame: pd.DataFrame,
) -> MarketSnapshot:
    price = _finite(quote.get("price"), _latest_price(frame))
    previous_close = _finite(quote.get("pricePreviousClose"), price)
    if price <= 0 or previous_close <= 0:
        raise ValueError("Microsoft Finance 没有有效市场价格。")
    timestamp = quote_timestamp(quote, instrument)
    session_status, delayed_seconds = _session_state(
        timestamp, instrument.market, instrument.timezone
    )
    opening = _finite(quote.get("priceDayOpen"), _finite(frame["Price"].iloc[0], previous_close))
    high = _finite(quote.get("priceDayHigh"), _finite(frame["Price"].max(), price))
    low = _finite(quote.get("priceDayLow"), _finite(frame["Price"].min(), price))
    volume = _finite(quote.get("accumulatedVolume"), _finite(frame["Volume"].sum()))
    change = _finite(quote.get("priceChange"), price - previous_close)
    asset_type = str(quote.get("securityType") or instrument.asset_type).lower()
    if asset_type == "fund" and instrument.asset_type == "etf":
        asset_type = "etf"
    exchange = str(
        quote.get("exchangeName")
        or quote.get("sourceExchangeName")
        or instrument.exchange
    )
    return MarketSnapshot(
        market=instrument.market,
        symbol=instrument.symbol,
        name=str(quote.get("displayName") or instrument.name),
        currency=str(quote.get("currency") or instrument.currency),
        price=round(price, 4),
        previous_close=round(previous_close, 4),
        change=round(change, 4),
        change_pct=round(change / previous_close, 6),
        open=round(opening, 4),
        high=round(high, 4),
        low=round(low, 4),
        volume=round(volume, 0),
        amount=round(price * volume, 2),
        timestamp=timestamp.isoformat(),
        session_status=session_status,
        provider="msn-finance-live",
        delayed_seconds=delayed_seconds,
        exchange=exchange,
        country=str(quote.get("country") or instrument.country),
        asset_type=asset_type,
        asset_type_label=ASSET_TYPE_LABELS.get(asset_type, asset_type.upper()),
        timezone=instrument.timezone,
        source_url=instrument.source_url,
    )


def _msn_realtime(source: str, symbol: str) -> RealtimeMarketData:
    instrument = resolve_msn_instrument(symbol, source)
    quote, intraday = fetch_msn_intraday(instrument)
    snapshot = _msn_snapshot(instrument, quote, intraday)
    return RealtimeMarketData(snapshot=snapshot, intraday=intraday)


def fetch_realtime_market(source: str, symbol: str) -> RealtimeMarketData:
    normalized_source = source.strip().lower()
    if normalized_source == "a-share":
        snapshot = _tencent_quote(symbol, "a-share")
        intraday = _tencent_intraday(symbol, "a-share")
        return RealtimeMarketData(snapshot=snapshot, intraday=intraday)
    if normalized_source in {"hk", "hong-kong"}:
        try:
            snapshot = _tencent_quote(symbol, "hk")
            intraday = _tencent_intraday(symbol, "hk")
            return RealtimeMarketData(snapshot=snapshot, intraday=intraday)
        except Exception as tencent_error:
            try:
                return _msn_realtime("hk", symbol)
            except Exception as msn_error:
                raise RuntimeError(
                    "港股实时行情不可用：腾讯（"
                    + summarize_error(tencent_error)
                    + "）；Microsoft Finance（"
                    + summarize_error(msn_error)
                    + "）"
                ) from msn_error
    if normalized_source in {"nasdaq", "us", "usa"}:
        try:
            result = _yahoo_intraday_result(symbol)
            intraday = _yahoo_intraday_frame(result)
            snapshot = _yahoo_snapshot(result, intraday)
            return RealtimeMarketData(snapshot=snapshot, intraday=intraday)
        except Exception as yahoo_error:
            try:
                result, asset_class = _nasdaq_intraday_result(symbol)
                intraday = _nasdaq_intraday_frame(result, asset_class)
                snapshot = _nasdaq_snapshot(result, intraday)
                return RealtimeMarketData(snapshot=snapshot, intraday=intraday)
            except Exception as nasdaq_error:
                try:
                    return _msn_realtime("nasdaq", symbol)
                except Exception as msn_error:
                    raise RuntimeError(
                        "美股实时行情不可用：Yahoo（"
                        + summarize_error(yahoo_error)
                        + "）；Nasdaq（"
                        + summarize_error(nasdaq_error)
                        + "）；Microsoft Finance（"
                        + summarize_error(msn_error)
                        + "）"
                    ) from msn_error
    if normalized_source in {"global", "world", "international"}:
        return _msn_realtime("global", symbol)
    raise ValueError("实时行情目前支持 a-share、nasdaq、hk 与 global。")


def intraday_records(frame: pd.DataFrame, limit: int = 500) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("分时数据条数必须为正整数。")
    result: list[dict[str, Any]] = []
    for timestamp, row in frame.tail(limit).iterrows():
        result.append(
            {
                "time": pd.Timestamp(timestamp).isoformat(),
                "price": round(_finite(row.get("Price")), 6),
                "average_price": round(_finite(row.get("AveragePrice")), 6),
                "volume": round(_finite(row.get("Volume")), 2),
                "cumulative_volume": round(
                    _finite(row.get("CumulativeVolume")), 2
                ),
                "amount": round(_finite(row.get("Amount")), 2),
            }
        )
    return result
