from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .numeric import clamp, finite_float
from .scoring import (
    DirectionBand,
    calibrate_direction,
    direction_band,
    direction_to_rating,
)


GAMMA_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
POLYMARKET_URL = "https://polymarket.com"
POLYMARKET_API_DOCS_URL = "https://docs.polymarket.com/market-data/overview"
POLYMARKET_PRICE_DOCS_URL = "https://docs.polymarket.com/concepts/prices-orderbook"
POLYMARKET_CACHE_SCHEMA = 1
MAX_RESPONSE_BYTES = 3_000_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36 RavenWatchAgentsPro/1.10"
)


_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple",),
    "AMZN": ("Amazon",),
    "GOOG": ("Google", "Alphabet"),
    "GOOGL": ("Google", "Alphabet"),
    "META": ("Meta", "Facebook"),
    "MSFT": ("Microsoft",),
    "NVDA": ("Nvidia",),
    "TSLA": ("Tesla",),
    "300750": ("CATL",),
    "600519": ("Kweichow Moutai", "Moutai"),
    "0700": ("Tencent",),
    "9988": ("Alibaba",),
}

_GENERIC_COMPANY_WORDS = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "plc",
    "stock",
}

_POSITIVE_IMPACT_PHRASES = (
    "avoid recession",
    "no recession",
    "rate cut",
    "cut interest rates",
    "lower interest rates",
    "peace deal",
    "ceasefire",
    "tariff reduction",
    "lift sanctions",
    "earnings beat",
    "beat earnings",
    "above expectations",
    "approval",
    "approved",
    "surpass",
    "exceed",
    "reach a market cap",
    "largest company",
    "record high",
    "gdp growth above",
    "economic growth above",
)

_NEGATIVE_IMPACT_PHRASES = (
    "rate hike",
    "raise interest rates",
    "no rate cut",
    "recession",
    "default",
    "bankrupt",
    "bankruptcy",
    "trade war",
    "military conflict",
    "war with",
    "new tariff",
    "impose tariff",
    "tariffs on",
    "sanction",
    "export ban",
    "import ban",
    "antitrust breakup",
    "break up",
    "earnings miss",
    "miss earnings",
    "below expectations",
    "fall below",
    "drop below",
    "market crash",
    "gdp growth below",
    "economic growth below",
)


@dataclass(frozen=True)
class SearchTopic:
    query: str
    kind: str
    terms: tuple[str, ...]
    relevance: float


@dataclass(frozen=True)
class PolymarketMarket:
    market_id: str
    event_id: str
    event_title: str
    question: str
    event_slug: str
    market_slug: str
    url: str
    yes_probability: float
    probability_source: str
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    volume: float
    volume_24h: float
    liquidity: float
    end_date: str
    updated_at: str
    resolution_source: str
    source_query: str
    relevance_kind: str
    relevance_score: float
    matched_terms: tuple[str, ...]
    impact_sign: int
    impact_label: str
    impact_trigger: str
    impact_confidence: float
    directional_score: float
    quality_score: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolymarketMarket:
        return cls(
            market_id=str(payload.get("market_id") or ""),
            event_id=str(payload.get("event_id") or ""),
            event_title=str(payload.get("event_title") or ""),
            question=str(payload.get("question") or ""),
            event_slug=str(payload.get("event_slug") or ""),
            market_slug=str(payload.get("market_slug") or ""),
            url=str(payload.get("url") or POLYMARKET_URL),
            yes_probability=float(payload.get("yes_probability") or 0.0),
            probability_source=str(payload.get("probability_source") or "unknown"),
            best_bid=finite_float(payload.get("best_bid")),
            best_ask=finite_float(payload.get("best_ask")),
            spread=finite_float(payload.get("spread")),
            volume=float(payload.get("volume") or 0.0),
            volume_24h=float(payload.get("volume_24h") or 0.0),
            liquidity=float(payload.get("liquidity") or 0.0),
            end_date=str(payload.get("end_date") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            resolution_source=str(payload.get("resolution_source") or ""),
            source_query=str(payload.get("source_query") or ""),
            relevance_kind=str(payload.get("relevance_kind") or "unknown"),
            relevance_score=float(payload.get("relevance_score") or 0.0),
            matched_terms=tuple(str(item) for item in payload.get("matched_terms") or ()),
            impact_sign=int(payload.get("impact_sign") or 0),
            impact_label=str(payload.get("impact_label") or "方向未判定"),
            impact_trigger=str(payload.get("impact_trigger") or ""),
            impact_confidence=float(payload.get("impact_confidence") or 0.0),
            directional_score=float(payload.get("directional_score") or 0.0),
            quality_score=float(payload.get("quality_score") or 0.0),
            explanation=str(payload.get("explanation") or ""),
        )


@dataclass(frozen=True)
class PolymarketSnapshot:
    market: str
    symbol: str
    company_name: str
    query_terms: tuple[str, ...]
    items: tuple[PolymarketMarket, ...]
    provider_status: str
    source_mode: str
    stale: bool
    cache_age_seconds: int
    fetched_at: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "query_terms": list(self.query_terms),
            "items": [item.to_dict() for item in self.items],
            "provider_status": self.provider_status,
            "provider": {
                "id": "polymarket-gamma",
                "label": "Polymarket Gamma API",
                "status": self.provider_status,
                "source_mode": self.source_mode,
                "source_url": GAMMA_SEARCH_URL,
                "documentation_url": POLYMARKET_API_DOCS_URL,
            },
            "source_mode": self.source_mode,
            "stale": self.stale,
            "cache_age_seconds": self.cache_age_seconds,
            "fetched_at": self.fetched_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolymarketSnapshot:
        provider = payload.get("provider")
        nested_status = provider.get("status") if isinstance(provider, dict) else None
        return cls(
            market=str(payload.get("market") or ""),
            symbol=str(payload.get("symbol") or ""),
            company_name=str(payload.get("company_name") or ""),
            query_terms=tuple(str(item) for item in payload.get("query_terms") or ()),
            items=tuple(
                PolymarketMarket.from_dict(item)
                for item in payload.get("items") or ()
                if isinstance(item, dict)
            ),
            provider_status=str(payload.get("provider_status") or nested_status or "cached"),
            source_mode=str(payload.get("source_mode") or "cache-stale"),
            stale=bool(payload.get("stale", True)),
            cache_age_seconds=int(payload.get("cache_age_seconds") or 0),
            fetched_at=str(payload.get("fetched_at") or ""),
            warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        )


JsonFetcher = Callable[[str, float], dict[str, Any]]


def _utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_words(value: Any) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value or "").casefold())
    return tuple(word for word in words if word not in _GENERIC_COMPANY_WORDS)


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def _concise_error(error: Exception, limit: int = 220) -> str:
    message = re.sub(r"\s+", " ", str(error)).strip() or error.__class__.__name__
    return message if len(message) <= limit else message[: limit - 1].rstrip() + "…"


def _search_topics(market: str, symbol: str, company_name: str) -> tuple[SearchTopic, ...]:
    normalized_symbol = symbol.strip().upper()
    base_symbol = normalized_symbol.split(".", 1)[0]
    aliases = list(
        _SYMBOL_ALIASES.get(normalized_symbol, _SYMBOL_ALIASES.get(base_symbol, ()))
    )
    company_words = _normalized_words(company_name)
    if company_words:
        aliases.append(" ".join(company_words[:4]))
    if normalized_symbol and not normalized_symbol.isdigit():
        aliases.append(normalized_symbol)
    direct_queries = _unique(aliases)[:2]
    topics = [
        SearchTopic(
            query=query,
            kind="direct",
            terms=_unique(list(_normalized_words(query))),
            relevance=0.95,
        )
        for query in direct_queries
        if len(query) >= 2
    ]
    macro_topics: dict[str, tuple[SearchTopic, ...]] = {
        "a-share": (
            SearchTopic(
                "China economy",
                "macro",
                ("china", "chinese", "gdp", "economy", "economic"),
                0.55,
            ),
            SearchTopic(
                "China tariffs",
                "macro",
                ("china", "chinese", "tariff", "trade war", "export ban"),
                0.50,
            ),
        ),
        "hk": (
            SearchTopic(
                "China economy",
                "macro",
                ("china", "chinese", "gdp", "economy", "economic"),
                0.55,
            ),
            SearchTopic(
                "China tariffs",
                "macro",
                ("china", "chinese", "tariff", "trade war", "export ban"),
                0.50,
            ),
        ),
        "nasdaq": (
            SearchTopic(
                "Federal Reserve interest rates",
                "macro",
                ("federal reserve", "fed", "interest rate", "rate cut", "rate hike"),
                0.50,
            ),
            SearchTopic(
                "US recession",
                "macro",
                ("recession", "us economy", "united states economy"),
                0.50,
            ),
        ),
        "global": (
            SearchTopic(
                "Federal Reserve interest rates",
                "macro",
                ("federal reserve", "fed", "interest rate", "rate cut", "rate hike"),
                0.45,
            ),
            SearchTopic(
                "US recession",
                "macro",
                ("recession", "us economy", "united states economy"),
                0.45,
            ),
        ),
    }
    topics.extend(macro_topics.get(market, ()))
    return tuple(topics[:4])


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _bounded_probability(value: Any) -> float | None:
    number = finite_float(value)
    if number is None or number < 0 or number > 1:
        return None
    return number


def _yes_probability(market: dict[str, Any]) -> tuple[float | None, str]:
    outcomes = [str(item).strip().casefold() for item in _json_array(market.get("outcomes"))]
    prices = _json_array(market.get("outcomePrices"))
    if "yes" not in outcomes:
        return None, "non-binary"
    yes_index = outcomes.index("yes")
    outcome_price = (
        _bounded_probability(prices[yes_index]) if yes_index < len(prices) else None
    )
    best_bid = _bounded_probability(market.get("bestBid"))
    best_ask = _bounded_probability(market.get("bestAsk"))
    last_trade = _bounded_probability(market.get("lastTradePrice"))
    if best_bid is not None and best_ask is not None and best_ask >= best_bid:
        spread = best_ask - best_bid
        if spread <= 0.10:
            return (best_bid + best_ask) / 2.0, "orderbook-midpoint"
        if last_trade is not None:
            return last_trade, "last-trade-wide-spread"
    if outcome_price is not None:
        return outcome_price, "gamma-outcome-price"
    if last_trade is not None:
        return last_trade, "last-trade"
    return None, "unavailable"


def _impact_direction(text: str) -> tuple[int, str, str, float]:
    normalized = re.sub(r"\s+", " ", text.casefold())
    positive = [phrase for phrase in _POSITIVE_IMPACT_PHRASES if phrase in normalized]
    negative = [phrase for phrase in _NEGATIVE_IMPACT_PHRASES if phrase in normalized]
    if "avoid recession" in positive or "no recession" in positive:
        negative = [item for item in negative if item != "recession"]
    if "no rate cut" in negative:
        positive = [item for item in positive if item != "rate cut"]
    if len(positive) > len(negative):
        return 1, "事件发生偏多", positive[0], min(1.0, 0.72 + 0.08 * len(positive))
    if len(negative) > len(positive):
        return -1, "事件发生偏空", negative[0], min(1.0, 0.72 + 0.08 * len(negative))
    return 0, "方向未判定", "", 0.0


def _relevance(
    text: str,
    topic: SearchTopic,
) -> tuple[float, tuple[str, ...], str]:
    normalized = re.sub(r"\s+", " ", text.casefold())
    matched = tuple(term for term in topic.terms if term and term in normalized)
    if topic.kind == "direct":
        meaningful = tuple(term for term in matched if len(term) >= 3)
        if not meaningful:
            return 0.0, (), "unrelated"
        bonus = min(0.04, max(0, len(meaningful) - 1) * 0.02)
        return min(1.0, topic.relevance + bonus), meaningful, "direct"
    if not matched:
        return 0.0, (), "unrelated"
    bonus = min(0.08, max(0, len(matched) - 1) * 0.02)
    return min(0.72, topic.relevance + bonus), matched, "macro"


def _log_quality(value: float, full_quality_at: float) -> float:
    if value <= 0:
        return 0.0
    return clamp(
        math.log1p(value) / math.log1p(full_quality_at),
        0.0,
        1.0,
    )


def _market_quality(
    *,
    relevance: float,
    impact_confidence: float,
    liquidity: float,
    volume: float,
    spread: float | None,
) -> float:
    liquidity_quality = _log_quality(liquidity, 100_000)
    volume_quality = _log_quality(volume, 1_000_000)
    spread_quality = 0.45 if spread is None else 1.0 - clamp(spread / 0.20, 0.0, 1.0)
    quality = 100 * (
        0.38 * relevance
        + 0.18 * impact_confidence
        + 0.16 * liquidity_quality
        + 0.13 * volume_quality
        + 0.10 * spread_quality
        + 0.05
    )
    return round(clamp(quality, 0.0, 100.0), 2)


def _number(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = finite_float(payload.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _safe_slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9-]", "", str(value or "").casefold())[:160]


def _market_from_payload(
    event: dict[str, Any],
    market: dict[str, Any],
    topic: SearchTopic,
    now: datetime,
) -> PolymarketMarket | None:
    if bool(event.get("closed")) or bool(market.get("closed")):
        return None
    if event.get("active") is False or market.get("active") is False:
        return None
    end_date = str(market.get("endDateIso") or market.get("endDate") or event.get("endDate") or "")
    parsed_end = _parse_datetime(end_date)
    if parsed_end is not None and parsed_end < now:
        return None
    probability, probability_source = _yes_probability(market)
    if probability is None:
        return None
    event_title = str(event.get("title") or "").strip()
    question = str(market.get("question") or event_title).strip()
    description = str(market.get("description") or event.get("description") or "")
    evidence_text = " ".join((event_title, question, description))
    relevance, matched_terms, relevance_kind = _relevance(evidence_text, topic)
    if relevance <= 0:
        return None
    impact_sign, impact_label, impact_trigger, impact_confidence = _impact_direction(
        evidence_text
    )
    best_bid = _bounded_probability(market.get("bestBid"))
    best_ask = _bounded_probability(market.get("bestAsk"))
    spread = finite_float(market.get("spread"))
    if spread is None and best_bid is not None and best_ask is not None:
        spread = max(0.0, best_ask - best_bid)
    if spread is not None:
        spread = clamp(spread, 0.0, 1.0)
    volume = _number(market, "volumeNum", "volume")
    volume_24h = _number(market, "volume24hr", "volume24hrClob")
    liquidity = _number(market, "liquidityNum", "liquidity", "liquidityClob")
    quality_score = _market_quality(
        relevance=relevance,
        impact_confidence=impact_confidence,
        liquidity=liquidity,
        volume=volume,
        spread=spread,
    )
    directional_score = (
        clamp((probability - 0.5) * 200.0 * impact_sign, -100.0, 100.0)
        if impact_sign
        else 0.0
    )
    event_slug = _safe_slug(event.get("slug"))
    market_slug = _safe_slug(market.get("slug"))
    url = f"{POLYMARKET_URL}/event/{event_slug}" if event_slug else POLYMARKET_URL
    probability_text = f"Yes 隐含概率 {probability:.1%}"
    if impact_sign:
        explanation = (
            f"{probability_text}；识别为{impact_label}（语义：{impact_trigger}）；"
            f"{ '直接关联' if relevance_kind == 'direct' else '宏观关联' }；"
            f"质量 {quality_score:.0f}/100。"
        )
    else:
        explanation = (
            f"{probability_text}；事件对股票方向的语义不够明确，仅展示而不计入多空评分；"
            f"质量 {quality_score:.0f}/100。"
        )
    return PolymarketMarket(
        market_id=str(market.get("id") or market.get("conditionId") or ""),
        event_id=str(event.get("id") or ""),
        event_title=event_title,
        question=question,
        event_slug=event_slug,
        market_slug=market_slug,
        url=url,
        yes_probability=round(probability, 6),
        probability_source=probability_source,
        best_bid=round(best_bid, 6) if best_bid is not None else None,
        best_ask=round(best_ask, 6) if best_ask is not None else None,
        spread=round(spread, 6) if spread is not None else None,
        volume=round(volume, 2),
        volume_24h=round(volume_24h, 2),
        liquidity=round(liquidity, 2),
        end_date=end_date,
        updated_at=str(market.get("updatedAt") or event.get("updatedAt") or ""),
        resolution_source=str(
            market.get("resolutionSource") or event.get("resolutionSource") or ""
        ),
        source_query=topic.query,
        relevance_kind=relevance_kind,
        relevance_score=round(relevance, 6),
        matched_terms=matched_terms,
        impact_sign=impact_sign,
        impact_label=impact_label,
        impact_trigger=impact_trigger,
        impact_confidence=round(impact_confidence, 6),
        directional_score=round(directional_score, 2),
        quality_score=quality_score,
        explanation=explanation,
    )


def _fetch_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Polymarket 响应超过大小限制。")
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Polymarket 返回格式无效。")
    return parsed


def _default_cache_dir() -> Path:
    configured = os.getenv("RAVENWATCHAGENTSPRO_POLYMARKET_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "RavenWatchAgentsPro" / "polymarket"
    return Path.home() / ".ravenwatchagentspro" / "polymarket"


def _cache_path(cache_dir: Path, market: str, symbol: str) -> Path:
    safe_market = re.sub(r"[^a-z0-9-]", "", market.casefold()) or "market"
    safe_symbol = re.sub(r"[^A-Z0-9._-]", "_", symbol.upper())[:48] or "SYMBOL"
    return cache_dir / f"{safe_market}-{safe_symbol}.json"


def _write_cache(snapshot: PolymarketSnapshot, path: Path, now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": POLYMARKET_CACHE_SCHEMA,
        "saved_at": now.isoformat(timespec="seconds"),
        "snapshot": snapshot.to_dict(),
    }
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def _read_cache(path: Path) -> tuple[PolymarketSnapshot | None, str]:
    if not path.exists():
        return None, ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if int(payload.get("schema_version") or 0) != POLYMARKET_CACHE_SCHEMA:
            raise ValueError("缓存版本不兼容")
        snapshot_payload = payload.get("snapshot")
        if not isinstance(snapshot_payload, dict):
            raise ValueError("缓存缺少快照")
        snapshot = PolymarketSnapshot.from_dict(snapshot_payload)
        return snapshot, ""
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"Polymarket 本地缓存不可读：{exc}"


def _cache_age_seconds(snapshot: PolymarketSnapshot, now: datetime) -> int:
    fetched = _parse_datetime(snapshot.fetched_at)
    if fetched is None:
        return 2_147_483_647
    return max(0, round((now - fetched).total_seconds()))


def _cached_snapshot(
    snapshot: PolymarketSnapshot,
    *,
    now: datetime,
    source_mode: str,
    warning: str = "",
) -> PolymarketSnapshot:
    age = _cache_age_seconds(snapshot, now)
    warnings = list(snapshot.warnings)
    if warning:
        warnings.append(warning)
    return replace(
        snapshot,
        provider_status="cached",
        source_mode=source_mode,
        stale=source_mode != "cache-fresh",
        cache_age_seconds=age,
        warnings=_unique(warnings),
    )


def _unavailable_snapshot(
    market: str,
    symbol: str,
    company_name: str,
    topics: tuple[SearchTopic, ...],
    now: datetime,
    *,
    source_mode: str,
    warnings: list[str],
) -> PolymarketSnapshot:
    return PolymarketSnapshot(
        market=market,
        symbol=symbol,
        company_name=company_name,
        query_terms=tuple(topic.query for topic in topics),
        items=(),
        provider_status="unavailable",
        source_mode=source_mode,
        stale=False,
        cache_age_seconds=0,
        fetched_at=now.isoformat(timespec="seconds"),
        warnings=_unique(warnings),
    )


def fetch_polymarket_snapshot(
    market: str,
    symbol: str,
    *,
    company_name: str = "",
    limit: int = 8,
    timeout_seconds: float = 10,
    force: bool = False,
    offline: bool = False,
    cache_dir: Path | None = None,
    fresh_seconds: int = 300,
    now: datetime | None = None,
    fetch_json: JsonFetcher | None = None,
) -> PolymarketSnapshot:
    current = _utc_now(now)
    normalized_market = market.strip().lower()
    normalized_symbol = symbol.strip().upper()
    topics = _search_topics(normalized_market, normalized_symbol, company_name)
    resolved_cache_dir = cache_dir or _default_cache_dir()
    path = _cache_path(resolved_cache_dir, normalized_market, normalized_symbol)
    cached, cache_warning = _read_cache(path)
    if cached is not None:
        age = _cache_age_seconds(cached, current)
        if offline:
            return _cached_snapshot(
                cached,
                now=current,
                source_mode="cache-offline",
                warning="离线模式：使用本地 Polymarket 快照。",
            )
        if not force and age <= fresh_seconds:
            return _cached_snapshot(cached, now=current, source_mode="cache-fresh")
    if offline:
        warnings = [cache_warning] if cache_warning else []
        warnings.append("离线模式下尚无该标的的 Polymarket 本地快照。")
        return _unavailable_snapshot(
            normalized_market,
            normalized_symbol,
            company_name,
            topics,
            current,
            source_mode="offline-empty",
            warnings=warnings,
        )

    loader = fetch_json or _fetch_json
    candidates: dict[str, PolymarketMarket] = {}
    warnings = [cache_warning] if cache_warning else []
    successful_queries = 0
    per_query_timeout = max(2.0, timeout_seconds / max(1, len(topics)))
    for topic in topics:
        query = urlencode(
            {
                "q": topic.query,
                "limit_per_type": max(3, min(10, limit)),
                "events_status": "active",
                "keep_closed_markets": 0,
                "search_tags": "false",
                "search_profiles": "false",
            }
        )
        url = f"{GAMMA_SEARCH_URL}?{query}"
        try:
            payload = loader(url, per_query_timeout)
            events = payload.get("events") or []
            if not isinstance(events, list):
                raise ValueError("events 字段格式无效")
            successful_queries += 1
        except Exception as exc:
            warnings.append(
                f"Polymarket 查询“{topic.query}”失败：{_concise_error(exc)}"
            )
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            markets = event.get("markets") or []
            if not isinstance(markets, list):
                continue
            for market_payload in markets:
                if not isinstance(market_payload, dict):
                    continue
                parsed = _market_from_payload(event, market_payload, topic, current)
                if parsed is None:
                    continue
                key = parsed.market_id or f"{parsed.event_id}:{parsed.question.casefold()}"
                previous = candidates.get(key)
                if previous is None or (
                    parsed.relevance_score,
                    parsed.quality_score,
                ) > (
                    previous.relevance_score,
                    previous.quality_score,
                ):
                    candidates[key] = parsed
    if successful_queries == 0:
        message = "；".join(warnings[-2:]) or "Polymarket 在线查询失败。"
        if cached is not None:
            return _cached_snapshot(
                cached,
                now=current,
                source_mode="cache-stale",
                warning="在线更新失败，已回退本地快照：" + message,
            )
        warnings.append("Polymarket 当前不可用，预测市场证据未纳入本次研判。")
        return _unavailable_snapshot(
            normalized_market,
            normalized_symbol,
            company_name,
            topics,
            current,
            source_mode="unavailable",
            warnings=warnings,
        )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item.relevance_score,
            item.quality_score,
            item.volume_24h,
            item.volume,
        ),
        reverse=True,
    )[: max(1, min(20, limit))]
    snapshot = PolymarketSnapshot(
        market=normalized_market,
        symbol=normalized_symbol,
        company_name=company_name,
        query_terms=tuple(topic.query for topic in topics),
        items=tuple(ranked),
        provider_status="ok" if successful_queries == len(topics) else "partial",
        source_mode="live",
        stale=False,
        cache_age_seconds=0,
        fetched_at=current.isoformat(timespec="seconds"),
        warnings=_unique(warnings),
    )
    try:
        _write_cache(snapshot, path, current)
    except OSError as exc:
        snapshot = replace(
            snapshot,
            warnings=snapshot.warnings + (f"Polymarket 快照写入失败：{exc}",),
        )
    return snapshot


def _freshness_factor(snapshot: dict[str, Any]) -> float:
    mode = str(snapshot.get("source_mode") or "unavailable")
    age = max(0.0, finite_float(snapshot.get("cache_age_seconds")) or 0.0)
    if mode == "live":
        return 1.0
    if mode == "cache-fresh":
        return 0.95
    if mode in {"cache-stale", "cache-offline"}:
        return max(0.35, 0.90 * math.exp(-age / (7 * 86_400)))
    return 0.0


def _assessment_label(score: float) -> str:
    return {
        DirectionBand.STRONG_BULLISH: "预测市场明显偏多",
        DirectionBand.BULLISH: "预测市场略偏多",
        DirectionBand.NEUTRAL: "预测市场中性",
        DirectionBand.BEARISH: "预测市场略偏空",
        DirectionBand.STRONG_BEARISH: "预测市场风险偏高",
    }[direction_band(score)]


def assess_polymarket(snapshot: PolymarketSnapshot | dict[str, Any]) -> dict[str, Any]:
    payload = snapshot.to_dict() if isinstance(snapshot, PolymarketSnapshot) else snapshot
    items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    weights: list[float] = []
    usable_indices: list[int] = []
    for item in items:
        relevance = clamp(finite_float(item.get("relevance_score")) or 0.0, 0.0, 1.0)
        quality = clamp(finite_float(item.get("quality_score")) or 0.0, 0.0, 100.0)
        impact_confidence = clamp(
            finite_float(item.get("impact_confidence")) or 0.0,
            0.0,
            1.0,
        )
        direction = clamp(
            finite_float(item.get("directional_score")) or 0.0,
            -100.0,
            100.0,
        )
        included = (
            int(item.get("impact_sign") or 0) != 0
            and relevance >= 0.40
            and quality >= 30
            and impact_confidence >= 0.50
        )
        evidence_weight = (
            quality / 100 * relevance * impact_confidence if included else 0.0
        )
        row = dict(item)
        row.update(
            {
                "included": included,
                "evidence_weight": round(evidence_weight, 6),
                "effective_weight": 0.0,
                "contribution": 0.0,
                "exclusion_reason": (
                    ""
                    if included
                    else "方向语义、关联度或市场质量不足，仅作背景展示。"
                ),
            }
        )
        rows.append(row)
        if included:
            usable_indices.append(len(rows) - 1)
            weights.append(evidence_weight)

    denominator = sum(weights)
    if denominator:
        raw_direction = sum(
            float(rows[index]["directional_score"])
            * float(rows[index]["evidence_weight"])
            for index in usable_indices
        ) / denominator
    else:
        raw_direction = 0.0
    raw_direction = clamp(raw_direction, -100.0, 100.0)
    weighted_deviation = 0.0
    for index in usable_indices:
        effective_weight = float(rows[index]["evidence_weight"]) / denominator
        contribution = float(rows[index]["directional_score"]) * effective_weight
        rows[index]["effective_weight"] = round(effective_weight, 6)
        rows[index]["contribution"] = round(contribution, 2)
        weighted_deviation += effective_weight * abs(
            float(rows[index]["directional_score"]) - raw_direction
        )
    agreement = clamp(1.0 - weighted_deviation / 100.0, 0.0, 1.0)
    usable = [rows[index] for index in usable_indices]
    breadth = min(1.0, len(usable) / 3.0)
    mean_quality = (
        sum(float(item["quality_score"]) for item in usable) / len(usable) / 100
        if usable
        else 0.0
    )
    direct_ratio = (
        sum(item.get("relevance_kind") == "direct" for item in usable) / len(usable)
        if usable
        else 0.0
    )
    freshness = _freshness_factor(payload)
    reliability = clamp(
        (
            0.35 * mean_quality
            + 0.25 * agreement
            + 0.25 * breadth
            + 0.15 * direct_ratio
        )
        * freshness,
        0.0,
        1.0,
    )
    calibrated = calibrate_direction(
        raw_direction,
        reliability,
        minimum_calibration=0.20,
    )
    directional_score = calibrated.directional_score
    score = direction_to_rating(directional_score)
    confidence = round(100 * reliability)
    available = bool(usable)
    source_mode = str(payload.get("source_mode") or "unavailable")
    cache_age = max(0, round(finite_float(payload.get("cache_age_seconds")) or 0.0))
    if available:
        summary = (
            f"纳入 {len(usable)} 个可定向预测市场，原始方向 {raw_direction:+.1f}，"
            f"一致度 {agreement:.0%}；经质量与新鲜度校准后为 {directional_score:+.1f}，"
            f"可信度 {confidence}%。"
        )
    else:
        summary = "未找到同时满足关联度、方向语义和市场质量门槛的预测市场，本项不参与综合评分。"
    process = [
        {
            "step": 1,
            "title": "发现与去重",
            "formula": "按公司别名、代码及市场宏观主题检索；同一 market_id 只保留最高关联结果",
            "result": f"发现 {len(rows)} 个市场，{len(usable)} 个通过定向门槛",
        },
        {
            "step": 2,
            "title": "概率与方向映射",
            "formula": "方向 = (Yes 概率 - 50%) × 200 × 事件影响符号",
            "result": f"加权前原始方向 {raw_direction:+.2f}",
        },
        {
            "step": 3,
            "title": "市场质量加权",
            "formula": "权重 = 质量 × 关联度 × 影响语义置信度",
            "result": f"平均质量 {mean_quality:.0%}，直接关联占比 {direct_ratio:.0%}",
        },
        {
            "step": 4,
            "title": "一致度与时效",
            "formula": "可靠度 = (质量35% + 一致度25% + 广度25% + 直接关联15%) × 新鲜度",
            "result": f"一致度 {agreement:.0%}，新鲜度 {freshness:.0%}，可靠度 {reliability:.0%}",
        },
        {
            "step": 5,
            "title": "向中性校准",
            "formula": "校准方向 = 原始方向 × [20% + 80% × 可靠度]；评级 = 50 + 方向/2",
            "result": f"校准系数 {calibrated.calibration_factor:.3f}，评级 {score:.2f}/100",
        },
    ]
    return {
        "available": available,
        "score": round(score, 2),
        "directional_score": round(directional_score, 2),
        "raw_directional_score": round(raw_direction, 2),
        "signal_strength": round(calibrated.signal_strength, 2),
        "label": _assessment_label(directional_score) if available else "预测市场证据不足",
        "confidence": confidence,
        "agreement": round(agreement, 6),
        "calibration_factor": round(calibrated.calibration_factor, 6),
        "market_count": len(rows),
        "included_market_count": len(usable),
        "direct_market_count": sum(
            item.get("relevance_kind") == "direct" for item in usable
        ),
        "source_mode": source_mode,
        "stale": bool(payload.get("stale")),
        "cache_age_seconds": cache_age,
        "markets": rows,
        "summary": summary,
        "process": process,
        "warnings": list(payload.get("warnings") or []),
        "limitations": [
            "预测市场价格反映参与者定价，不等于事实或必然结果。",
            "事件对个股的多空映射采用保守规则；语义不明确的市场不计分。",
            "宏观市场只作为低相关证据，不能替代公司基本面与价格验证。",
        ],
        "method": {
            "provider": "Polymarket Gamma API",
            "probability_reference": POLYMARKET_PRICE_DOCS_URL,
            "api_reference": POLYMARKET_API_DOCS_URL,
            "minimum_quality": 30,
            "minimum_relevance": 0.40,
            "minimum_impact_confidence": 0.50,
            "source_mode": source_mode,
        },
    }
