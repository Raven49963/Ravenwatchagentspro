from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import re
import threading
import time
from typing import Any
from urllib import parse as urlparse
from urllib import error as urlerror
from urllib import request as urlrequest
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .metadata import PRODUCT_USER_AGENT


MSN_SEARCH_URL = (
    "https://services.bingapis.com/"
    "contentservices-finance.csautosuggest/api/v1/Query"
)
MSN_FINANCE_URL = "https://assets.msn.com/service/Finance"
# Public browser key embedded by Microsoft on MSN Money pages. It can be
# overridden without rebuilding if Microsoft rotates it.
MSN_PUBLIC_API_KEY = os.getenv(
    "RAVENWATCHAGENTS_MSN_API_KEY",
    "0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM",
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    f"AppleWebKit/537.36 Chrome/126.0 Safari/537.36 {PRODUCT_USER_AGENT}"
)

SECURITY_TYPES = {
    "ST": "stock",
    "FE": "etf",
    "FO": "fund",
    "XI": "index",
    "ET": "etf",
    "RE": "reit",
}
ASSET_TYPE_LABELS = {
    "stock": "股票",
    "etf": "ETF",
    "fund": "基金",
    "index": "指数",
    "reit": "REIT",
}
COUNTRY_CURRENCIES = {
    "AU": "AUD",
    "AT": "EUR",
    "BE": "EUR",
    "CA": "CAD",
    "CH": "CHF",
    "CN": "CNY",
    "DE": "EUR",
    "DK": "DKK",
    "ES": "EUR",
    "FI": "EUR",
    "FR": "EUR",
    "GB": "GBP",
    "HK": "HKD",
    "IN": "INR",
    "IT": "EUR",
    "JP": "JPY",
    "KR": "KRW",
    "NL": "EUR",
    "NO": "NOK",
    "SE": "SEK",
    "SG": "SGD",
    "TW": "TWD",
    "US": "USD",
}
COUNTRY_TIMEZONES = {
    "AU": "Australia/Sydney",
    "AT": "Europe/Vienna",
    "BE": "Europe/Brussels",
    "CA": "America/Toronto",
    "CH": "Europe/Zurich",
    "CN": "Asia/Shanghai",
    "DE": "Europe/Berlin",
    "DK": "Europe/Copenhagen",
    "ES": "Europe/Madrid",
    "FI": "Europe/Helsinki",
    "FR": "Europe/Paris",
    "GB": "Europe/London",
    "HK": "Asia/Hong_Kong",
    "IN": "Asia/Kolkata",
    "IT": "Europe/Rome",
    "JP": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    "NL": "Europe/Amsterdam",
    "NO": "Europe/Oslo",
    "SE": "Europe/Stockholm",
    "SG": "Asia/Singapore",
    "TW": "Asia/Taipei",
    "US": "America/New_York",
}
SUFFIX_HINTS = {
    ".AS": ("NL", {"XAMS"}),
    ".AX": ("AU", {"XASX"}),
    ".BO": ("IN", {"XBOM"}),
    ".BR": ("BE", {"XBRU"}),
    ".CO": ("DK", {"XCSE"}),
    ".DE": ("DE", {"XETR", "XFRA"}),
    ".HE": ("FI", {"XHEL"}),
    ".HK": ("HK", {"XHKG"}),
    ".KS": ("KR", {"XKRX", "XKOS"}),
    ".L": ("GB", {"XLON"}),
    ".MC": ("ES", {"XMAD"}),
    ".MI": ("IT", {"XMIL"}),
    ".NS": ("IN", {"XNSE"}),
    ".OL": ("NO", {"XOSL"}),
    ".PA": ("FR", {"XPAR"}),
    ".SI": ("SG", {"XSES"}),
    ".ST": ("SE", {"XSTO"}),
    ".SW": ("CH", {"XSWX"}),
    ".T": ("JP", {"XTKS"}),
    ".TO": ("CA", {"XTSE"}),
    ".TW": ("TW", {"XTAI", "ROCO"}),
    ".VI": ("AT", {"XWBO"}),
}
COUNTRY_SUFFIXES = {
    "AU": ".AX",
    "AT": ".VI",
    "BE": ".BR",
    "CA": ".TO",
    "CH": ".SW",
    "DE": ".DE",
    "DK": ".CO",
    "ES": ".MC",
    "FI": ".HE",
    "FR": ".PA",
    "GB": ".L",
    "HK": ".HK",
    "IN": ".NS",
    "IT": ".MI",
    "JP": ".T",
    "KR": ".KS",
    "NL": ".AS",
    "NO": ".OL",
    "SE": ".ST",
    "SG": ".SI",
    "TW": ".TW",
}


@dataclass(frozen=True)
class MSNInstrument:
    symbol: str
    name: str
    instrument_id: str
    full_instrument: str
    market: str
    exchange: str
    mic: str
    country: str
    asset_type: str
    currency: str
    timezone: str
    description: str = ""

    @property
    def label(self) -> str:
        asset_label = ASSET_TYPE_LABELS.get(self.asset_type, self.asset_type.upper())
        return f"{self.name} · {self.symbol} · {self.exchange} · {asset_label}"

    @property
    def source_url(self) -> str:
        return msn_instrument_url(self.instrument_id)

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["label"] = self.label
        payload["asset_type_label"] = ASSET_TYPE_LABELS.get(
            self.asset_type, self.asset_type.upper()
        )
        payload["source_url"] = self.source_url
        return payload


_cache_lock = threading.RLock()
_search_cache: dict[tuple[str, str, str], tuple[float, tuple[MSNInstrument, ...]]] = {}
_resolve_cache: dict[tuple[str, str], tuple[float, MSNInstrument]] = {}


def msn_instrument_url(instrument_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9]", "", instrument_id)
    return f"https://www.msn.com/en-us/money/stockdetails/fi-{safe_id}"


def _read_json(url: str, *, timeout: float = 20, attempts: int = 3) -> Any:
    request = urlrequest.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.8",
            "Origin": "https://www.msn.com",
            "Referer": "https://www.msn.com/",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                body = response.read(5_000_001)
            break
        except urlerror.HTTPError as exc:
            last_error = exc
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt + 1 >= attempts:
                raise
            time.sleep(0.35 * (attempt + 1))
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.35 * (attempt + 1))
    else:
        assert last_error is not None
        raise last_error
    if len(body) > 5_000_000:
        raise ValueError("Microsoft Finance response is too large.")
    return json.loads(body.decode("utf-8", errors="replace"))


def _market_for_country(country: str) -> str:
    if country == "CN":
        return "a-share"
    if country == "US":
        return "nasdaq"
    if country == "HK":
        return "hk"
    return "global"


def _canonical_symbol(raw_symbol: str, country: str, mic: str) -> str:
    raw = raw_symbol.strip().upper()
    if country == "CN" and raw.isdigit():
        return raw.zfill(6)
    if country == "HK" and raw.isdigit():
        number = str(int(raw))
        return f"{number.zfill(4) if len(number) <= 4 else number}.HK"
    if country == "US":
        return raw
    suffix = COUNTRY_SUFFIXES.get(country, "")
    if country == "IN" and mic == "XBOM":
        suffix = ".BO"
    return raw if not suffix or raw.endswith(suffix) else raw + suffix


def _parse_search_item(raw_item: Any) -> MSNInstrument | None:
    try:
        item = json.loads(raw_item) if isinstance(raw_item, str) else dict(raw_item)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    raw_symbol = str(item.get("RT00S") or item.get("OS001") or "").strip()
    instrument_id = str(item.get("SecId") or "").strip()
    country = str(item.get("RT0EC") or "").strip().upper()
    if not raw_symbol or not instrument_id or not country:
        return None
    mic = str(item.get("ExMicCode") or item.get("LS01Z") or "").strip().upper()
    exchange = str(item.get("AC040") or mic or country).strip()
    security_code = str(item.get("OS010") or "").strip().upper()
    asset_type = SECURITY_TYPES.get(security_code, "stock")
    name = str(
        item.get("OS01W")
        or item.get("RT0SN")
        or item.get("FriendlyName")
        or raw_symbol
    ).strip()
    return MSNInstrument(
        symbol=_canonical_symbol(raw_symbol, country, mic),
        name=name,
        instrument_id=instrument_id,
        full_instrument=str(item.get("FullInstrument") or ""),
        market=_market_for_country(country),
        exchange=exchange,
        mic=mic,
        country=country,
        asset_type=asset_type,
        currency=COUNTRY_CURRENCIES.get(country, ""),
        timezone=COUNTRY_TIMEZONES.get(country, "UTC"),
        description=str(item.get("Description") or "")[:500],
    )


def _market_matches(instrument: MSNInstrument, market: str) -> bool:
    normalized = market.strip().lower()
    if normalized in {"us", "usa"}:
        normalized = "nasdaq"
    if normalized in {"world", "international"}:
        normalized = "global"
    if normalized == "global":
        return instrument.market == "global"
    return instrument.market == normalized


def search_msn_instruments(
    query: str,
    *,
    market: str = "",
    asset_type: str = "",
    limit: int = 12,
) -> tuple[MSNInstrument, ...]:
    cleaned = re.sub(r"\s+", " ", query).strip()[:80]
    if len(cleaned) < 1 or any(ord(character) < 32 for character in cleaned):
        return ()
    normalized_market = market.strip().lower()
    normalized_asset = asset_type.strip().lower()
    key = (cleaned.casefold(), normalized_market, normalized_asset)
    now = time.monotonic()
    with _cache_lock:
        cached = _search_cache.get(key)
        if cached is not None and now < cached[0]:
            return cached[1][: max(1, min(limit, 30))]

    params = urlparse.urlencode(
        {"query": cleaned, "market": "en-us", "count": 30}
    )
    payload = _read_json(f"{MSN_SEARCH_URL}?{params}")
    raw_items = ((payload or {}).get("data") or {}).get("stocks") or []
    parsed = [item for raw in raw_items if (item := _parse_search_item(raw))]
    if normalized_market:
        parsed = [item for item in parsed if _market_matches(item, normalized_market)]
    if normalized_asset and normalized_asset != "all":
        parsed = [item for item in parsed if item.asset_type == normalized_asset]

    query_base = _comparison_symbol(cleaned)

    def rank(item: MSNInstrument) -> tuple[int, int, str]:
        item_base = _comparison_symbol(item.symbol)
        exact = 0 if item_base == query_base else 1
        starts = 0 if item_base.startswith(query_base) else 1
        return exact, starts, item.name.casefold()

    parsed.sort(key=rank)
    unique: list[MSNInstrument] = []
    seen: set[str] = set()
    for item in parsed:
        if item.instrument_id in seen:
            continue
        seen.add(item.instrument_id)
        unique.append(item)
    result = tuple(unique)
    with _cache_lock:
        _search_cache[key] = (now + 300, result)
    return result[: max(1, min(limit, 30))]


def _symbol_hint(symbol: str) -> tuple[str, str, set[str]]:
    normalized = symbol.strip().upper()
    for suffix, (country, mics) in sorted(
        SUFFIX_HINTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)], country, mics
    return normalized, "", set()


def _comparison_symbol(symbol: str) -> str:
    base, _country, _mics = _symbol_hint(symbol)
    comparable = re.sub(r"[^A-Z0-9]", "", base.upper()).lstrip("0")
    return comparable or "0"


def resolve_msn_instrument(
    symbol: str,
    market: str,
    *,
    name_hint: str = "",
) -> MSNInstrument:
    normalized_symbol = symbol.strip().upper()
    normalized_market = market.strip().lower()
    if normalized_market in {"us", "usa"}:
        normalized_market = "nasdaq"
    key = (normalized_market, normalized_symbol)
    now = time.monotonic()
    with _cache_lock:
        cached = _resolve_cache.get(key)
        if cached is not None and now < cached[0]:
            return cached[1]

    base, expected_country, expected_mics = _symbol_hint(normalized_symbol)
    if normalized_market == "a-share":
        base = "".join(character for character in base if character.isdigit()).zfill(6)
        expected_country = "CN"
    elif normalized_market == "hk":
        digits = "".join(character for character in base if character.isdigit())
        if not digits:
            raise ValueError("港股代码格式无效。")
        base = str(int(digits))
        expected_country = "HK"
        expected_mics = {"XHKG"}
    elif normalized_market == "nasdaq":
        expected_country = "US"

    queries = [base]
    if name_hint.strip():
        queries.insert(0, f"{base} {name_hint.strip()[:50]}")
    if normalized_market == "hk":
        queries.extend((f"{base} Hong Kong", f"{base} HKEX"))

    candidates: list[MSNInstrument] = []
    for query in dict.fromkeys(queries):
        candidates.extend(search_msn_instruments(query, limit=30))
        if any(
            item.country == expected_country
            and (not expected_mics or item.mic in expected_mics)
            for item in candidates
        ):
            break

    def score(item: MSNInstrument) -> tuple[int, int, int, str]:
        item_base = _comparison_symbol(item.symbol)
        wanted_base = _comparison_symbol(base)
        country_penalty = 0 if not expected_country or item.country == expected_country else 3
        mic_penalty = 0 if not expected_mics or item.mic in expected_mics else 2
        symbol_penalty = 0 if item_base == wanted_base else 1
        return country_penalty, symbol_penalty, mic_penalty, item.name.casefold()

    if not candidates:
        raise ValueError(f"Microsoft Finance 未找到证券：{symbol}")
    candidates.sort(key=score)
    selected = candidates[0]
    penalties = score(selected)[:3]
    if penalties[0] >= 3 or penalties[1] > 0:
        raise ValueError(f"Microsoft Finance 未找到匹配市场的证券：{symbol}")
    with _cache_lock:
        _resolve_cache[key] = (now + 3_600, selected)
    return selected


def _finance_request(path: str, **params: Any) -> Any:
    query = {
        "apikey": MSN_PUBLIC_API_KEY,
        "activityId": str(uuid4()),
        "ocid": "finance-utils-peregrine",
        "cm": "en-us",
        "it": "web",
        "scn": "ANON",
        "wrapodata": "false",
        **params,
    }
    return _read_json(
        f"{MSN_FINANCE_URL}/{path}?{urlparse.urlencode(query)}",
        timeout=25,
    )


def fetch_msn_quote(instrument: MSNInstrument) -> dict[str, Any]:
    payload = _finance_request("Quotes", ids=instrument.instrument_id)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Microsoft Finance 未返回报价。")
    quote = payload[0]
    if not isinstance(quote, dict) or not quote.get("price"):
        raise ValueError("Microsoft Finance 报价缺少有效价格。")
    return quote


def _range_code(start: str, end: str) -> str:
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    if days <= 130:
        return "3M"
    if days <= 230:
        return "1Y"
    if days <= 500:
        return "1Y"
    if days <= 1_250:
        return "3Y"
    if days <= 2_100:
        return "5Y"
    return "5Y"


def fetch_msn_chart(
    instrument: MSNInstrument,
    chart_type: str,
) -> dict[str, Any]:
    payload = _finance_request(
        "Charts",
        ids=instrument.instrument_id,
        type=chart_type,
        chartflag=0,
    )
    if not isinstance(payload, list) or not payload:
        raise ValueError("Microsoft Finance 未返回图表数据。")
    chart = payload[0]
    if not isinstance(chart, dict) or not isinstance(chart.get("series"), dict):
        raise ValueError("Microsoft Finance 图表结构无效。")
    return chart


def _series_frame(chart: dict[str, Any], timezone_name: str) -> pd.DataFrame:
    series = chart.get("series") or {}
    timestamps = series.get("timeStamps") or []
    values = {
        "Open": series.get("openPrices") or [],
        "High": series.get("pricesHigh") or [],
        "Low": series.get("pricesLow") or [],
        "Close": series.get("prices") or [],
        "Volume": series.get("volumes") or [],
    }
    length = len(timestamps)
    if length < 2 or any(len(items) != length for items in values.values()):
        raise ValueError("Microsoft Finance 图表缺少完整 OHLCV 序列。")
    index = pd.to_datetime(timestamps, utc=True, errors="coerce")
    try:
        index = index.tz_convert(timezone_name)
    except (KeyError, TypeError, ValueError):
        index = index.tz_convert("UTC")
    frame = pd.DataFrame(values, index=index.tz_localize(None))
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame["Volume"] = frame["Volume"].fillna(0).clip(lower=0)
    frame = frame.loc[
        (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    ]
    if frame.empty:
        raise ValueError("Microsoft Finance 图表没有可用行情行。")
    frame.index.name = "date"
    return frame


def fetch_msn_ohlcv(
    symbol: str,
    market: str,
    start: str,
    end: str,
    *,
    name_hint: str = "",
) -> pd.DataFrame:
    instrument = resolve_msn_instrument(symbol, market, name_hint=name_hint)
    chart = fetch_msn_chart(instrument, _range_code(start, end))
    frame = _series_frame(chart, instrument.timezone)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() + pd.offsets.Day(1)
    frame = frame.loc[(frame.index >= start_ts) & (frame.index < end_ts)]
    if frame.empty:
        raise ValueError("Microsoft Finance 未返回指定日期范围内的行情。")
    frame.attrs.update(
        provider="msn-finance",
        provider_endpoint="assets.msn.com",
        provider_url=instrument.source_url,
        instrument=instrument.to_dict(),
    )
    return frame


def fetch_msn_intraday(
    instrument: MSNInstrument,
) -> tuple[dict[str, Any], pd.DataFrame]:
    quote = fetch_msn_quote(instrument)
    chart = fetch_msn_chart(instrument, "1D1M")
    frame = _series_frame(chart, instrument.timezone)
    close = frame["Close"].rename("Price")
    volume = frame["Volume"].rename("Volume")
    cumulative_volume = volume.cumsum().rename("CumulativeVolume")
    amount = (close * volume).cumsum().rename("Amount")
    average = amount.div(cumulative_volume.replace(0, np.nan)).ffill().fillna(
        close.expanding().mean()
    ).rename("AveragePrice")
    intraday = pd.concat(
        [close, average, volume, cumulative_volume, amount], axis=1
    )
    intraday.attrs.update(
        provider="msn-finance-live",
        market=instrument.market,
        symbol=instrument.symbol,
        instrument=instrument.to_dict(),
    )
    return quote, intraday


def quote_timestamp(quote: dict[str, Any], instrument: MSNInstrument) -> pd.Timestamp:
    raw = quote.get("timeLastTraded") or quote.get("timeLastUpdated")
    timestamp = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp(datetime.now(timezone.utc))
    try:
        return timestamp.tz_convert(ZoneInfo(instrument.timezone))
    except (KeyError, ValueError):
        return timestamp
