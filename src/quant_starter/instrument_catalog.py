from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import argparse
import base64
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Iterable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .data import normalize_hk_symbol
from .symbols import presets_for_source


CATALOG_SCHEMA_VERSION = 1
CATALOG_REFRESH_SECONDS = 24 * 60 * 60
CATALOG_STALE_SECONDS = 14 * 24 * 60 * 60
CATALOG_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
    "RavenWatchAgentsPro/1.9"
)
CATALOG_RESOURCE = Path(__file__).with_name("resources") / "instrument_catalog.json.gz"

EASTMONEY_ENDPOINT = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_SOURCE_URLS = {
    "a-share": "https://quote.eastmoney.com/center/gridlist.html#hs_a_board",
    "hk": "https://quote.eastmoney.com/center/gridlist.html#hk_stocks",
}
EASTMONEY_FILTERS = {
    "a-share": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
    "hk": "m:128+t:3,m:128+t:4",
}
NASDAQ_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NASDAQ_SOURCE_URL = "https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs"
SINA_A_COUNT_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeStockCount?node=hs_a"
)
SINA_A_LIST_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
SINA_HK_LIST_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHKStockData"
)
SINA_SOURCE_URLS = {
    "a-share": "https://vip.stock.finance.sina.com.cn/mkt/#hs_a",
    "hk": "https://vip.stock.finance.sina.com.cn/mkt/#qbgg_hk",
}

ASSET_TYPE_LABELS = {
    "stock": "股票",
    "etf": "ETF",
    "fund": "基金",
    "index": "指数",
    "reit": "REIT",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timestamp(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return 0.0


def _read_bytes_with_windows_http(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
) -> bytes:
    script = (
        "$ProgressPreference='SilentlyContinue';"
        "$response=Invoke-WebRequest -UseBasicParsing -Uri $env:RAVENWATCHAGENTSPRO_CATALOG_URL "
        "-Headers @{'User-Agent'='Mozilla/5.0';'Referer'='https://quote.eastmoney.com/'} "
        f"-TimeoutSec {max(5, int(timeout))};"
        "$bytes=[Text.Encoding]::UTF8.GetBytes([string]$response.Content);"
        "[Console]::Out.Write([Convert]::ToBase64String($bytes))"
    )
    environment = os.environ.copy()
    environment["RAVENWATCHAGENTSPRO_CATALOG_URL"] = url
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        timeout=timeout + 10,
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    body = base64.b64decode(completed.stdout.strip(), validate=True)
    if len(body) > max_bytes:
        raise ValueError("证券目录响应超过大小限制。")
    return body


def _read_bytes(url: str, *, timeout: float = 25, max_bytes: int = 20_000_000) -> bytes:
    windows_first = os.name == "nt" and urlparse.urlsplit(url).hostname == "push2.eastmoney.com"
    last_error: BaseException | None = None
    if windows_first:
        try:
            return _read_bytes_with_windows_http(
                url, timeout=timeout, max_bytes=max_bytes
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            last_error = exc
    request = urlrequest.Request(
        url,
        headers={
            "User-Agent": CATALOG_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            "Referer": "https://quote.eastmoney.com/",
        },
    )
    openers: tuple[Callable[..., Any], ...] = (
        urlrequest.urlopen,
        urlrequest.build_opener(urlrequest.ProxyHandler({})).open,
    )
    for attempt in range(3):
        for opener in openers:
            try:
                with opener(request, timeout=timeout) as response:
                    body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise ValueError("证券目录响应超过大小限制。")
                return body
            except (OSError, TimeoutError, urlerror.URLError, urlerror.HTTPError) as exc:
                last_error = exc
        if attempt < 2:
            time.sleep(0.4 * (attempt + 1))
    if os.name == "nt" and not windows_first:
        try:
            return _read_bytes_with_windows_http(
                url, timeout=timeout, max_bytes=max_bytes
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class CatalogInstrument:
    market: str
    symbol: str
    name: str
    asset_type: str
    exchange: str
    country: str
    currency: str
    category: str
    rank: int = 0
    market_cap: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name} · {self.symbol}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label"] = self.label
        payload["asset_type_label"] = ASSET_TYPE_LABELS.get(
            self.asset_type, self.asset_type.upper()
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CatalogInstrument:
        return cls(
            market=str(payload.get("market") or ""),
            symbol=str(payload.get("symbol") or "").upper(),
            name=str(payload.get("name") or payload.get("symbol") or ""),
            asset_type=str(payload.get("asset_type") or "stock").lower(),
            exchange=str(payload.get("exchange") or ""),
            country=str(payload.get("country") or ""),
            currency=str(payload.get("currency") or ""),
            category=str(payload.get("category") or "未分类"),
            rank=int(payload.get("rank") or 0),
            market_cap=float(payload.get("market_cap") or 0),
        )


@dataclass(frozen=True)
class MarketCatalog:
    market: str
    items: tuple[CatalogInstrument, ...]
    provider: str
    provider_label: str
    provider_url: str
    updated_at: str
    source_mode: str
    warning: str = ""

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "provider_url": self.provider_url,
            "updated_at": self.updated_at,
            "items": [asdict(item) for item in self.items],
        }

    @classmethod
    def from_storage_dict(
        cls,
        payload: dict[str, Any],
        *,
        source_mode: str,
    ) -> MarketCatalog:
        return cls(
            market=str(payload.get("market") or ""),
            items=tuple(
                CatalogInstrument.from_dict(item)
                for item in payload.get("items", [])
                if isinstance(item, dict)
            ),
            provider=str(payload.get("provider") or "catalog-snapshot"),
            provider_label=str(payload.get("provider_label") or "证券目录快照"),
            provider_url=str(payload.get("provider_url") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            source_mode=source_mode,
        )


def _a_share_exchange(code: str) -> tuple[str, str]:
    if code.startswith(("4", "8", "92")):
        return "北交所", "北交所股票"
    if code.startswith("688"):
        return "上交所", "科创板"
    if code.startswith("6"):
        return "上交所", "沪市主板"
    if code.startswith("3"):
        return "深交所", "创业板"
    return "深交所", "深市主板"


def _eastmoney_page(market: str, page: int, page_size: int) -> dict[str, Any]:
    params = urlparse.urlencode(
        {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f20",
            "fs": EASTMONEY_FILTERS[market],
            "fields": "f12,f14,f13,f20,f100",
        }
    )
    payload = json.loads(_read_bytes(f"{EASTMONEY_ENDPOINT}?{params}").decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise RuntimeError("东方财富证券目录未返回有效数据。")
    return data


def fetch_eastmoney_catalog(market: str) -> MarketCatalog:
    if market not in EASTMONEY_FILTERS:
        raise ValueError(f"东方财富目录不支持市场：{market}")
    page_size = 1_000
    first = _eastmoney_page(market, 1, page_size)
    total = max(0, int(first.get("total") or 0))
    rows = list(first.get("diff") or [])
    pages = max(1, math.ceil(total / page_size))
    for page in range(2, pages + 1):
        time.sleep(0.08)
        rows.extend(_eastmoney_page(market, page, page_size).get("diff") or [])

    items: list[CatalogInstrument] = []
    seen: set[str] = set()
    for rank, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        raw_code = str(raw.get("f12") or "").strip()
        name = str(raw.get("f14") or "").strip()
        if not raw_code or not name or name == "-":
            continue
        if market == "a-share":
            digits = "".join(character for character in raw_code if character.isdigit())
            if len(digits) != 6:
                continue
            symbol = digits
            exchange, board = _a_share_exchange(symbol)
            country, currency = "CN", "CNY"
            category = str(raw.get("f100") or board).strip() or board
        else:
            try:
                symbol = normalize_hk_symbol(raw_code)
            except ValueError:
                continue
            exchange, country, currency = "港交所", "HK", "HKD"
            category = str(raw.get("f100") or "香港股票").strip() or "香港股票"
        if symbol in seen:
            continue
        seen.add(symbol)
        try:
            market_cap = float(raw.get("f20") or 0)
        except (TypeError, ValueError):
            market_cap = 0.0
        items.append(
            CatalogInstrument(
                market=market,
                symbol=symbol,
                name=name,
                asset_type="stock",
                exchange=exchange,
                country=country,
                currency=currency,
                category=category,
                rank=rank,
                market_cap=market_cap,
            )
        )
    if len(items) < 100:
        raise RuntimeError(f"{market} 在线证券目录数量异常：{len(items)}")
    label = "东方财富 A 股目录" if market == "a-share" else "东方财富港股目录"
    return MarketCatalog(
        market=market,
        items=tuple(items),
        provider="eastmoney-catalog",
        provider_label=label,
        provider_url=EASTMONEY_SOURCE_URLS[market],
        updated_at=_utc_now(),
        source_mode="online",
    )


def _sina_page(market: str, page: int) -> Any:
    if market == "a-share":
        endpoint = SINA_A_LIST_URL
        params = {
            "page": page,
            "num": 100,
            "sort": "symbol",
            "asc": 1,
            "node": "hs_a",
            "symbol": "",
            "_s_r_a": "page",
        }
    elif market == "hk":
        endpoint = SINA_HK_LIST_URL
        params = {
            "page": page,
            "num": 60,
            "sort": "symbol",
            "asc": 1,
            "node": "qbgg_hk",
            "_s_r_a": "page",
        }
    else:
        raise ValueError(f"新浪目录不支持市场：{market}")
    url = f"{endpoint}?{urlparse.urlencode(params)}"
    return json.loads(_read_bytes(url).decode("utf-8", errors="replace"))


def fetch_sina_catalog(market: str) -> MarketCatalog:
    if market == "a-share":
        total = int(json.loads(_read_bytes(SINA_A_COUNT_URL).decode("utf-8")))
        page_count = max(1, math.ceil(total / 100))
    elif market == "hk":
        total = 0
        page_count = 100
    else:
        raise ValueError(f"新浪目录不支持市场：{market}")

    rows: list[dict[str, Any]] = []
    for page in range(1, page_count + 1):
        payload = _sina_page(market, page)
        if not isinstance(payload, list) or not payload:
            if market == "hk":
                break
            raise RuntimeError(f"新浪 {market} 目录第 {page} 页为空。")
        rows.extend(item for item in payload if isinstance(item, dict))
        time.sleep(0.04)

    items: list[CatalogInstrument] = []
    seen: set[str] = set()
    for rank, raw in enumerate(rows, start=1):
        name = str(raw.get("name") or "").strip()
        if market == "a-share":
            symbol = str(raw.get("code") or "").strip()
            if not re.fullmatch(r"\d{6}", symbol):
                continue
            exchange, category = _a_share_exchange(symbol)
            country, currency = "CN", "CNY"
        else:
            raw_symbol = str(raw.get("symbol") or "").strip()
            try:
                symbol = normalize_hk_symbol(raw_symbol)
            except ValueError:
                continue
            exchange, country, currency = "港交所", "HK", "HKD"
            category = "香港创业板" if raw_symbol.startswith("08") else "香港主板"
            if not name:
                name = str(raw.get("engname") or "").strip()
        if not name or symbol in seen:
            continue
        seen.add(symbol)
        try:
            market_cap = float(raw.get("mktcap") or 0)
        except (TypeError, ValueError):
            market_cap = 0.0
        items.append(
            CatalogInstrument(
                market=market,
                symbol=symbol,
                name=name,
                asset_type="stock",
                exchange=exchange,
                country=country,
                currency=currency,
                category=category,
                rank=rank,
                market_cap=market_cap,
            )
        )

    minimum = max(100, int(total * 0.9)) if market == "a-share" else 2_000
    if len(items) < minimum:
        raise RuntimeError(f"新浪 {market} 证券目录数量异常：{len(items)}")
    return MarketCatalog(
        market=market,
        items=tuple(items),
        provider="sina-finance-catalog",
        provider_label="新浪财经 A 股目录" if market == "a-share" else "新浪财经港股目录",
        provider_url=SINA_SOURCE_URLS[market],
        updated_at=_utc_now(),
        source_mode="online",
    )


def fetch_china_catalog(market: str) -> MarketCatalog:
    errors: list[str] = []
    for loader in (
        lambda: fetch_sina_catalog(market),
        lambda: fetch_eastmoney_catalog(market),
    ):
        try:
            return loader()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("；".join(errors))


US_EXCHANGES = {
    "Q": "NASDAQ Global Select",
    "G": "NASDAQ Global Market",
    "S": "NASDAQ Capital Market",
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}
US_EXCLUDED_NAME_PARTS = (
    " warrant",
    " warrants",
    " right",
    " rights",
    " unit",
    " units",
    " preferred stock",
    " preferred shares",
    " debt securities",
    " notes due",
    " bond due",
)


def _clean_us_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.replace("$", " ")).strip(" -")
    replacements = (
        (r"\s+-\s+Common Stock$", ""),
        (r"\s+Common Stock$", ""),
        (r"\s+-\s+Ordinary Shares$", ""),
        (r"\s+Ordinary Shares$", ""),
        (r"\s+-\s+American Depositary Shares$", " ADR"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned[:180].strip() or value[:180].strip()


def _us_asset_type(name: str, etf_flag: str) -> str | None:
    lowered = f" {name.casefold()}"
    if etf_flag.upper() == "Y":
        return "etf"
    if any(part in lowered for part in US_EXCLUDED_NAME_PARTS):
        return None
    if "real estate investment trust" in lowered or re.search(r"\breit\b", lowered):
        return "reit"
    return "stock"


def _parse_nasdaq_directory(
    body: bytes,
    *,
    listed_on_nasdaq: bool,
    start_rank: int = 0,
) -> list[CatalogInstrument]:
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    items: list[CatalogInstrument] = []
    for offset, raw in enumerate(reader, start=1):
        if listed_on_nasdaq:
            raw_symbol = str(raw.get("Symbol") or "").strip().upper()
            market_code = str(raw.get("Market Category") or "").strip().upper()
        else:
            raw_symbol = str(raw.get("ACT Symbol") or "").strip().upper()
            market_code = str(raw.get("Exchange") or "").strip().upper()
        if not raw_symbol or raw_symbol.startswith("FILE CREATION TIME"):
            continue
        if str(raw.get("Test Issue") or "").strip().upper() == "Y":
            continue
        symbol = raw_symbol.replace(".", "-")
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,13}", symbol):
            continue
        raw_name = str(raw.get("Security Name") or symbol).strip()
        asset_type = _us_asset_type(raw_name, str(raw.get("ETF") or ""))
        if asset_type is None:
            continue
        exchange = US_EXCHANGES.get(
            market_code, "NASDAQ" if listed_on_nasdaq else market_code
        )
        items.append(
            CatalogInstrument(
                market="nasdaq",
                symbol=symbol,
                name=_clean_us_name(raw_name),
                asset_type=asset_type,
                exchange=exchange,
                country="US",
                currency="USD",
                category=exchange,
                rank=start_rank + offset,
            )
        )
    return items


def fetch_nasdaq_trader_catalog() -> MarketCatalog:
    nasdaq = _parse_nasdaq_directory(
        _read_bytes(NASDAQ_DIRECTORY_URL), listed_on_nasdaq=True
    )
    other = _parse_nasdaq_directory(
        _read_bytes(OTHER_DIRECTORY_URL),
        listed_on_nasdaq=False,
        start_rank=len(nasdaq),
    )
    items: list[CatalogInstrument] = []
    seen: set[str] = set()
    for item in [*nasdaq, *other]:
        if item.symbol in seen:
            continue
        seen.add(item.symbol)
        items.append(item)
    if len(items) < 1_000:
        raise RuntimeError(f"Nasdaq Trader 证券目录数量异常：{len(items)}")
    return MarketCatalog(
        market="nasdaq",
        items=tuple(items),
        provider="nasdaq-trader-directory",
        provider_label="Nasdaq Trader 官方证券目录",
        provider_url=NASDAQ_SOURCE_URL,
        updated_at=_utc_now(),
        source_mode="online",
    )


def _preset_catalog(market: str) -> MarketCatalog:
    items = tuple(
        CatalogInstrument(
            market=item.source,
            symbol=item.symbol,
            name=item.name,
            asset_type=item.asset_type,
            exchange=item.exchange,
            country=item.country,
            currency=item.currency,
            category=item.category or "精选证券",
            rank=index,
        )
        for index, item in enumerate(presets_for_source(market), start=1)
    )
    return MarketCatalog(
        market=market,
        items=items,
        provider="built-in-presets",
        provider_label="Raven Watch Agents Pro 精选目录",
        provider_url="",
        updated_at=_utc_now(),
        source_mode="built-in",
    )


def _merge_presets(catalog: MarketCatalog) -> MarketCatalog:
    available = {item.symbol.upper(): item for item in catalog.items}
    preferred: list[CatalogInstrument] = []
    generic_categories = {"美国股票", "沪深股票", "香港股票", "精选证券"}
    for preset in _preset_catalog(catalog.market).items:
        current = available.pop(preset.symbol.upper(), None)
        if current is None:
            preferred.append(preset)
            continue
        category = (
            current.category
            if preset.category in generic_categories
            else preset.category or current.category
        )
        preferred.append(
            replace(
                current,
                name=preset.name or current.name,
                asset_type=preset.asset_type or current.asset_type,
                exchange=preset.exchange or current.exchange,
                country=preset.country or current.country,
                currency=preset.currency or current.currency,
                category=category,
            )
        )

    remaining = list(available.values())
    if any(item.market_cap > 0 for item in remaining):
        remaining.sort(
            key=lambda item: (
                0 if item.market_cap > 0 else 1,
                -item.market_cap,
                item.rank,
                item.symbol,
            )
        )
    else:
        remaining.sort(key=lambda item: (item.rank, item.symbol))
    merged = [*preferred, *remaining]
    return replace(
        catalog,
        items=tuple(replace(item, rank=index) for index, item in enumerate(merged, start=1)),
    )


PROVIDER_LOADERS: dict[str, Callable[[], MarketCatalog]] = {
    "a-share": lambda: fetch_china_catalog("a-share"),
    "nasdaq": fetch_nasdaq_trader_catalog,
    "hk": lambda: fetch_china_catalog("hk"),
}


def build_catalog_snapshot() -> dict[str, Any]:
    markets: dict[str, Any] = {}
    for market in ("a-share", "nasdaq", "hk"):
        markets[market] = _merge_presets(PROVIDER_LOADERS[market]()).to_storage_dict()
    markets["global"] = _preset_catalog("global").to_storage_dict()
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "markets": markets,
    }


def write_catalog_snapshot(payload: dict[str, Any], path: Path = CATALOG_RESOURCE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def _read_snapshot(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if int(payload.get("schema_version") or 0) != CATALOG_SCHEMA_VERSION:
        raise ValueError("证券目录快照版本不兼容。")
    if not isinstance(payload.get("markets"), dict):
        raise ValueError("证券目录快照缺少市场数据。")
    return payload


def _default_cache_dir() -> Path:
    configured = os.getenv("RAVENWATCHAGENTSPRO_CATALOG_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "RavenWatchAgentsPro" / "catalog"
    return Path.home() / ".ravenwatchagentspro" / "catalog"


class InstrumentCatalogService:
    def __init__(
        self,
        *,
        bundled_path: Path = CATALOG_RESOURCE,
        cache_dir: Path | None = None,
        auto_refresh: bool = True,
    ) -> None:
        self.bundled_path = bundled_path
        self.cache_dir = cache_dir or _default_cache_dir()
        self.auto_refresh = auto_refresh
        self._catalogs: dict[str, MarketCatalog] = {}
        self._bundled: dict[str, Any] | None = None
        self._refreshing: set[str] = set()
        self._lock = threading.RLock()

    def _bundled_market(self, market: str) -> MarketCatalog | None:
        if not self.bundled_path.exists():
            return None
        if self._bundled is None:
            self._bundled = _read_snapshot(self.bundled_path)
        payload = self._bundled.get("markets", {}).get(market)
        if not isinstance(payload, dict):
            return None
        return MarketCatalog.from_storage_dict(payload, source_mode="bundled")

    def _cache_path(self, market: str) -> Path:
        return self.cache_dir / f"{market}.json.gz"

    def _cached_market(self, market: str) -> MarketCatalog | None:
        path = self._cache_path(market)
        if not path.exists():
            return None
        try:
            payload = _read_snapshot(path).get("markets", {}).get(market)
            if isinstance(payload, dict):
                return MarketCatalog.from_storage_dict(payload, source_mode="cache")
        except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
            return None
        return None

    def _save_cache(self, catalog: MarketCatalog) -> None:
        payload = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "generated_at": catalog.updated_at,
            "markets": {catalog.market: catalog.to_storage_dict()},
        }
        try:
            write_catalog_snapshot(payload, self._cache_path(catalog.market))
        except OSError:
            pass

    def _initial_market(self, market: str) -> MarketCatalog:
        cached = self._cached_market(market)
        bundled = self._bundled_market(market)
        candidates = [item for item in (cached, bundled) if item is not None]
        if candidates:
            return max(candidates, key=lambda item: _timestamp(item.updated_at))
        return _preset_catalog(market)

    def _needs_refresh(self, catalog: MarketCatalog) -> bool:
        return (
            catalog.market in PROVIDER_LOADERS
            and time.time() - _timestamp(catalog.updated_at) > CATALOG_REFRESH_SECONDS
        )

    def _start_background_refresh(self, market: str) -> None:
        with self._lock:
            if market in self._refreshing:
                return
            self._refreshing.add(market)

        def run() -> None:
            try:
                self.refresh_market(market)
            finally:
                with self._lock:
                    self._refreshing.discard(market)

        threading.Thread(
            target=run,
            name=f"catalog-refresh-{market}",
            daemon=True,
        ).start()

    def get_market(self, market: str, *, force_refresh: bool = False) -> MarketCatalog:
        normalized = market.strip().lower()
        if normalized not in {"a-share", "nasdaq", "hk", "global"}:
            raise ValueError(f"未知证券市场：{market}")
        if force_refresh and normalized in PROVIDER_LOADERS:
            return self.refresh_market(normalized)
        with self._lock:
            catalog = self._catalogs.get(normalized)
            if catalog is None:
                catalog = self._initial_market(normalized)
                self._catalogs[normalized] = catalog
        if self.auto_refresh and self._needs_refresh(catalog):
            self._start_background_refresh(normalized)
        return catalog

    def refresh_market(self, market: str) -> MarketCatalog:
        normalized = market.strip().lower()
        loader = PROVIDER_LOADERS.get(normalized)
        if loader is None:
            return self.get_market(normalized)
        try:
            catalog = _merge_presets(loader())
        except Exception as exc:
            fallback = self._catalogs.get(normalized) or self._initial_market(normalized)
            age = time.time() - _timestamp(fallback.updated_at)
            suffix = "，目录可能已过期" if age > CATALOG_STALE_SECONDS else ""
            catalog = replace(
                fallback,
                warning=f"在线目录更新失败：{type(exc).__name__}: {exc}{suffix}",
            )
        else:
            self._save_cache(catalog)
        with self._lock:
            self._catalogs[normalized] = catalog
        return catalog

    def summary(self, market: str) -> dict[str, Any]:
        catalog = self.get_market(market)
        return {
            "count": len(catalog.items),
            "updated_at": catalog.updated_at,
            "provider": catalog.provider_label,
            "source_mode": catalog.source_mode,
        }

    def query(
        self,
        *,
        market: str,
        q: str = "",
        asset_type: str = "all",
        category: str = "all",
        page: int = 1,
        page_size: int = 50,
        refresh: bool = False,
    ) -> dict[str, Any]:
        catalog = self.get_market(market, force_refresh=refresh)
        normalized_asset = asset_type.strip().lower() or "all"
        if normalized_asset not in {"all", *ASSET_TYPE_LABELS}:
            raise ValueError(f"未知资产类型：{asset_type}")
        needle = re.sub(r"\s+", " ", q).strip().casefold()
        symbol_needles = {needle} if needle else set()
        if needle and catalog.market == "hk":
            try:
                symbol_needles.add(normalize_hk_symbol(needle).casefold())
            except ValueError:
                pass
        elif needle and catalog.market == "nasdaq":
            symbol_needles.add(needle.replace(".", "-"))
        selected_category = category.strip() or "all"

        asset_filtered = [
            item
            for item in catalog.items
            if normalized_asset == "all" or item.asset_type == normalized_asset
        ]
        searched = [
            item
            for item in asset_filtered
            if not needle
            or any(alias in item.symbol.casefold() for alias in symbol_needles)
            or needle in item.name.casefold()
            or needle in item.category.casefold()
            or needle in item.exchange.casefold()
        ]
        category_counts = Counter(item.category or "未分类" for item in searched)
        filtered = [
            item
            for item in searched
            if selected_category == "all" or item.category == selected_category
        ]
        if needle:
            def relevance(item: CatalogInstrument) -> tuple[int, int, int]:
                symbol = item.symbol.casefold()
                name = item.name.casefold()
                if symbol in symbol_needles:
                    score = 0
                elif any(symbol.startswith(alias) for alias in symbol_needles):
                    score = 1
                elif name.startswith(needle):
                    score = 2
                else:
                    score = 3
                return score, item.rank, len(item.name)

            filtered.sort(key=relevance)
        page_size = max(10, min(int(page_size), 100))
        pages = max(1, math.ceil(len(filtered) / page_size))
        page = max(1, min(int(page), pages))
        start = (page - 1) * page_size
        selected = filtered[start : start + page_size]
        asset_counts = Counter(item.asset_type for item in catalog.items)
        with self._lock:
            refreshing = catalog.market in self._refreshing
        return {
            "market": catalog.market,
            "query": q.strip(),
            "asset_type": normalized_asset,
            "category": selected_category,
            "catalog_total": len(catalog.items),
            "filtered_total": len(filtered),
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "items": [item.to_dict() for item in selected],
            "asset_counts": dict(sorted(asset_counts.items())),
            "category_counts": [
                {"category": name, "count": count}
                for name, count in sorted(
                    category_counts.items(), key=lambda entry: (-entry[1], entry[0])
                )
            ],
            "source": {
                "id": catalog.provider,
                "label": catalog.provider_label,
                "url": catalog.provider_url,
                "mode": catalog.source_mode,
                "updated_at": catalog.updated_at,
                "refreshing": refreshing,
                "warning": catalog.warning,
            },
        }


catalog_service = InstrumentCatalogService()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Raven Watch Agents Pro instrument catalog")
    parser.add_argument("--output", type=Path, default=CATALOG_RESOURCE)
    args = parser.parse_args()
    payload = build_catalog_snapshot()
    write_catalog_snapshot(payload, args.output)
    counts = {
        market: len(data.get("items", []))
        for market, data in payload["markets"].items()
    }
    print(json.dumps({"output": str(args.output), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
