from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
import ipaddress
import json
import math
import os
import queue
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36 RavenWatchAgentsPro/1.9"
)
SEC_USER_AGENT = os.environ.get(
    "RAVENWATCHAGENTSPRO_SEC_USER_AGENT",
    "RavenWatchAgentsPro/1.9 quant-research contact@example.com",
).strip()
VERIFICATION_SOURCE_THRESHOLD = 5
_gdelt_cache_lock = threading.RLock()
_gdelt_cache: dict[
    tuple[str, str], tuple[float, float, tuple["NewsArticle", ...]]
] = {}


@dataclass(frozen=True)
class NewsSourceReference:
    source_id: str
    title: str
    publisher: str
    published_at: str
    url: str
    provider: str
    provider_label: str
    source_kind: str = "media"
    credibility: str = "财经媒体"
    official: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsArticle:
    article_id: str
    title: str
    publisher: str
    published_at: str
    url: str
    provider: str
    provider_label: str
    summary: str = ""
    provider_url: str = ""
    source_kind: str = "media"
    credibility: str = "财经媒体"
    event_id: str = ""
    event_category: str = "company-update"
    verification_status: str = "single-source"
    verification_count: int = 1
    provider_count: int = 1
    official_source_count: int = 0
    verification_score: float = 0.0
    first_reported_at: str = ""
    last_reported_at: str = ""
    corroborating_sources: tuple[NewsSourceReference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsProviderStatus:
    provider: str
    label: str
    status: str
    item_count: int
    message: str = ""
    source_url: str = ""
    source_kind: str = "media"
    credibility: str = "财经媒体"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsVerification:
    required_sources: int = VERIFICATION_SOURCE_THRESHOLD
    raw_article_count: int = 0
    event_count: int = 0
    five_source_verified_count: int = 0
    official_primary_count: int = 0
    corroborated_count: int = 0
    independent_source_count: int = 0
    verified_event_ratio: float = 0.0
    methodology: str = (
        "同一事件按标题语义、事件类别与发布时间聚合；独立发布机构去重后达到 5 个才标记为五源验证。"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsFeed:
    market: str
    symbol: str
    query: str
    items: tuple[NewsArticle, ...]
    providers: tuple[NewsProviderStatus, ...]
    warnings: tuple[str, ...]
    fetched_at: str
    source_portals: tuple[dict[str, str], ...] = ()
    verification: NewsVerification = field(default_factory=NewsVerification)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
            "providers": [item.to_dict() for item in self.providers],
            "warnings": list(self.warnings),
            "fetched_at": self.fetched_at,
            "source_portals": list(self.source_portals),
            "verification": self.verification.to_dict(),
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _plain_text(value: Any, limit: int = 500) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(str(value or ""))
        text = " ".join(parser.parts)
    except Exception:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _safe_http_url(value: Any) -> str:
    candidate = str(value or "").strip().replace(" ", "%20")
    if not candidate or any(ord(character) < 32 for character in candidate):
        return ""
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").strip().lower()
        _ = parsed.port
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    if hostname == "localhost" or hostname.endswith(".local"):
        return ""
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            return ""
    except ValueError:
        pass
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _read_bytes(
    url: str,
    *,
    timeout_seconds: float = 8,
    max_bytes: int = MAX_RESPONSE_BYTES,
    user_agent: str = USER_AGENT,
) -> bytes:
    safe_url = _safe_http_url(url)
    if not safe_url:
        raise ValueError("新闻数据源地址无效。")
    request = Request(
        safe_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json, application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("新闻数据源响应超过大小限制。")
    return payload


def _published_iso(value: Any, *, offset_hours: int = 0) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1_000
        try:
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return parsed.isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            return ""

    raw = str(value).strip()
    if not raw:
        return ""
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            for pattern in (
                "%Y%m%dT%H%M%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M",
            ):
                try:
                    parsed = datetime.strptime(raw, pattern)
                    break
                except ValueError:
                    continue
    if parsed is None:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=offset_hours)))
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _publisher_from_url(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def _article(
    *,
    title: Any,
    publisher: Any,
    published: Any,
    url: Any,
    provider: str,
    provider_label: str,
    summary: Any = "",
    offset_hours: int = 0,
    provider_url: str = "",
    source_kind: str = "media",
    credibility: str = "财经媒体",
) -> NewsArticle | None:
    normalized_title = _plain_text(title, 220)
    normalized_url = _safe_http_url(url)
    if not normalized_title or not normalized_url:
        return None
    normalized_publisher = _plain_text(publisher, 80) or _publisher_from_url(
        normalized_url
    )
    identity = sha256(
        f"{normalized_title.casefold()}\n{normalized_url}".encode("utf-8")
    ).hexdigest()[:20]
    return NewsArticle(
        article_id=identity,
        title=normalized_title,
        publisher=normalized_publisher,
        published_at=_published_iso(published, offset_hours=offset_hours),
        url=normalized_url,
        provider=provider,
        provider_label=provider_label,
        summary=_plain_text(summary, 260),
        provider_url=_safe_http_url(provider_url),
        source_kind=source_kind,
        credibility=credibility,
    )


def _fetch_eastmoney_news(symbol: str, limit: int) -> list[NewsArticle]:
    akshare = __import__("akshare")
    frame = akshare.stock_news_em(symbol=symbol)
    if frame is None or frame.empty or "新闻标题" not in frame.columns:
        return []
    items: list[NewsArticle] = []
    for _, row in frame.head(max(limit * 2, limit)).iterrows():
        item = _article(
            title=row.get("新闻标题", ""),
            publisher=row.get("文章来源", ""),
            published=row.get("发布时间", ""),
            url=row.get("新闻链接", ""),
            provider="eastmoney",
            provider_label="东方财富（AKShare）",
            summary=row.get("新闻内容", ""),
            offset_hours=8,
            provider_url="https://finance.eastmoney.com/",
            source_kind="media",
            credibility="主流财经门户",
        )
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def _fetch_yahoo_news(symbol: str, limit: int, timeout_seconds: float) -> list[NewsArticle]:
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urlencode(
        {
            "s": symbol,
            "region": "US",
            "lang": "en-US",
        }
    )
    payload = _read_bytes(url, timeout_seconds=timeout_seconds)
    return _parse_rss_articles(
        payload,
        provider="yahoo-finance",
        provider_label="Yahoo Finance",
        limit=limit,
        provider_url="https://finance.yahoo.com/",
        source_kind="aggregator",
        credibility="国际财经门户",
    )


def _gdelt_query(symbol: str, company_name: str) -> str:
    candidates = [company_name.strip(), symbol.strip()]
    terms: list[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if not normalized or normalized.casefold() in {item.casefold() for item in terms}:
            continue
        safe_term = normalized.replace('"', "")[:80]
        if safe_term:
            terms.append(safe_term)
    quoted = [f'"{term}"' for term in terms[:2]]
    return f"({' OR '.join(quoted)})" if len(quoted) > 1 else quoted[0]


def _fetch_gdelt_news(
    symbol: str,
    company_name: str,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    cache_key = (symbol.strip().upper(), company_name.strip().casefold())
    now = time.monotonic()
    with _gdelt_cache_lock:
        cached = _gdelt_cache.get(cache_key)
        if cached and now < cached[0]:
            return list(cached[2])
    query = _gdelt_query(symbol, company_name)
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(
        {
            "query": query,
            "mode": "artlist",
            "maxrecords": min(250, max(50, limit * 4)),
            "timespan": "30d",
            "sort": "datedesc",
            "format": "json",
        }
    )
    try:
        payload = json.loads(_read_bytes(url, timeout_seconds=timeout_seconds))
    except Exception:
        with _gdelt_cache_lock:
            stale = _gdelt_cache.get(cache_key)
            if stale and now < stale[1]:
                return list(stale[2])
        raise
    items: list[NewsArticle] = []
    for raw in (payload or {}).get("articles") or []:
        article_url = raw.get("url") or raw.get("url_mobile") or ""
        publisher = raw.get("domain") or _publisher_from_url(str(article_url))
        country = _plain_text(raw.get("sourcecountry"), 40)
        language = _plain_text(raw.get("language"), 30)
        context = " · ".join(part for part in (country, language) if part)
        item = _article(
            title=raw.get("title"),
            publisher=publisher,
            published=raw.get("seendate"),
            url=article_url,
            provider="gdelt",
            provider_label="GDELT 全球新闻索引",
            summary=context,
            provider_url="https://www.gdeltproject.org/",
            source_kind="aggregator",
            credibility="跨媒体原始报道索引",
        )
        if item is not None:
            items.append(item)
        if len(items) >= min(250, max(50, limit * 4)):
            break
    with _gdelt_cache_lock:
        _gdelt_cache[cache_key] = (
            now + 10 * 60,
            now + 24 * 60 * 60,
            tuple(items),
        )
    return items


def _xml_child_text(element: ET.Element, names: set[str]) -> str:
    for child in element:
        local_name = child.tag.rsplit("}", 1)[-1].casefold()
        if local_name in names and child.text:
            return child.text.strip()
    return ""


def _parse_rss_articles(
    payload: bytes,
    *,
    provider: str,
    provider_label: str,
    limit: int,
    provider_url: str = "",
    source_kind: str = "media",
    credibility: str = "财经媒体",
) -> list[NewsArticle]:
    root = ET.fromstring(payload)
    items: list[NewsArticle] = []
    for raw in root.iter():
        if raw.tag.rsplit("}", 1)[-1].casefold() != "item":
            continue
        item = _article(
            title=_xml_child_text(raw, {"title"}),
            publisher=_xml_child_text(raw, {"source", "provider"}),
            published=_xml_child_text(raw, {"pubdate", "date"}),
            url=(
                _xml_child_text(raw, {"link"})
                or _xml_child_text(raw, {"guid"})
            ),
            provider=provider,
            provider_label=provider_label,
            summary=_xml_child_text(raw, {"description", "summary"}),
            provider_url=provider_url,
            source_kind=source_kind,
            credibility=credibility,
        )
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def _fetch_nasdaq_news(
    symbol: str,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    url = "https://www.nasdaq.com/feed/rssoutbound?" + urlencode(
        {"symbol": symbol}
    )
    payload = _read_bytes(url, timeout_seconds=timeout_seconds)
    return _parse_rss_articles(
        payload,
        provider="nasdaq-rss",
        provider_label="Nasdaq 官方 RSS",
        limit=limit,
        provider_url="https://www.nasdaq.com/nasdaq-rss-feeds",
        source_kind="exchange-media",
        credibility="交易所新闻频道",
    )


def _fetch_cninfo_announcements(symbol: str, limit: int) -> list[NewsArticle]:
    akshare = __import__("akshare")
    end = datetime.now().date()
    start = end - timedelta(days=210)
    frame = akshare.stock_zh_a_disclosure_report_cninfo(
        symbol=symbol,
        market="沪深京",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if frame is None or frame.empty or "公告标题" not in frame.columns:
        return []
    items: list[NewsArticle] = []
    for _, row in frame.head(max(limit * 2, limit)).iterrows():
        item = _article(
            title=row.get("公告标题", ""),
            publisher="巨潮资讯",
            published=row.get("公告时间", ""),
            url=row.get("公告链接", ""),
            provider="cninfo",
            provider_label="巨潮资讯公告",
            summary="上市公司信息披露公告",
            offset_hours=8,
            provider_url="https://www.cninfo.com.cn/",
            source_kind="filing",
            credibility="法定信息披露平台",
        )
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


_reference_cache_lock = threading.RLock()
_sec_ticker_cache: tuple[float, dict[str, tuple[int, str]]] = (0.0, {})
_hkex_stock_cache: tuple[float, dict[str, tuple[int, str]]] = (0.0, {})


def _sec_company_map() -> dict[str, tuple[int, str]]:
    global _sec_ticker_cache
    now = time.monotonic()
    with _reference_cache_lock:
        if now < _sec_ticker_cache[0]:
            return _sec_ticker_cache[1]
    payload = json.loads(
        _read_bytes(
            "https://www.sec.gov/files/company_tickers.json",
            timeout_seconds=8,
            user_agent=SEC_USER_AGENT,
        )
    )
    mapping: dict[str, tuple[int, str]] = {}
    for item in (payload or {}).values():
        ticker = str(item.get("ticker") or "").strip().upper()
        cik = int(item.get("cik_str") or 0)
        if ticker and cik:
            mapping[ticker] = (cik, str(item.get("title") or ticker))
    with _reference_cache_lock:
        _sec_ticker_cache = (now + 86_400, mapping)
    return mapping


def _fetch_sec_filings(
    symbol: str,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    lookup_symbol = symbol.replace("-", ".").upper()
    company = _sec_company_map().get(lookup_symbol) or _sec_company_map().get(
        symbol.upper()
    )
    if company is None:
        return []
    cik, company_name = company
    payload = json.loads(
        _read_bytes(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            timeout_seconds=timeout_seconds,
            user_agent=SEC_USER_AGENT,
        )
    )
    recent = ((payload or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    descriptions = recent.get("primaryDocDescription") or []
    accepted = recent.get("acceptanceDateTime") or []
    filed = recent.get("filingDate") or []
    accepted_forms = (
        "8-K",
        "10-Q",
        "10-K",
        "6-K",
        "20-F",
        "40-F",
        "DEF 14A",
        "SC 13",
        "S-1",
        "S-3",
        "424B",
        "N-PORT",
        "N-CEN",
        "497",
    )
    items: list[NewsArticle] = []
    row_count = min(
        len(forms), len(accessions), len(documents), max(len(accepted), len(filed))
    )
    for index in range(row_count):
        form = str(forms[index] or "")
        if not form.startswith(accepted_forms):
            continue
        accession = str(accessions[index] or "")
        document = str(documents[index] or "")
        if not accession or not document:
            continue
        accession_path = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_path}/{document}"
        )
        description = str(
            descriptions[index] if index < len(descriptions) else ""
        ).strip()
        published = (
            accepted[index]
            if index < len(accepted) and accepted[index]
            else filed[index]
            if index < len(filed)
            else ""
        )
        item = _article(
            title=f"{form} · {description or company_name}",
            publisher="U.S. SEC EDGAR",
            published=published,
            url=url,
            provider="sec-edgar",
            provider_label="SEC EDGAR 官方文件",
            summary=f"{company_name} 向美国证券交易委员会提交的 {form} 文件",
            provider_url="https://www.sec.gov/edgar/search/",
            source_kind="filing",
            credibility="监管机构官方",
        )
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


class _HKEXTitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self.row: dict[str, str] | None = None
        self.td_class = ""
        self.headline_depth = 0
        self.doc_link_depth = 0
        self.anchor_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag == "tr":
            self.row = {"published": "", "headline": "", "title": "", "url": ""}
        elif self.row is not None and tag == "td":
            self.td_class = " ".join(classes)
        elif self.row is not None and tag == "div":
            if self.headline_depth:
                self.headline_depth += 1
            elif "headline" in classes:
                self.headline_depth = 1
            if self.doc_link_depth:
                self.doc_link_depth += 1
            elif "doc-link" in classes:
                self.doc_link_depth = 1
        elif self.row is not None and tag == "a" and self.doc_link_depth:
            self.anchor_depth = 1
            self.row["url"] = urljoin(
                "https://www1.hkexnews.hk/", attributes.get("href", "")
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.anchor_depth:
            self.anchor_depth = 0
        elif tag == "td":
            self.td_class = ""
        elif tag == "div":
            if self.headline_depth:
                self.headline_depth -= 1
            if self.doc_link_depth:
                self.doc_link_depth -= 1
        elif tag == "tr" and self.row is not None:
            if self.row.get("url") and self.row.get("title"):
                self.rows.append(self.row)
            self.row = None
            self.td_class = ""
            self.headline_depth = 0
            self.doc_link_depth = 0
            self.anchor_depth = 0

    def handle_data(self, data: str) -> None:
        if self.row is None:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if "release-time" in self.td_class:
            self.row["published"] += " " + text
        if self.headline_depth:
            self.row["headline"] += " " + text
        if self.anchor_depth:
            self.row["title"] += " " + text


def _hkex_company_map() -> dict[str, tuple[int, str]]:
    global _hkex_stock_cache
    now = time.monotonic()
    with _reference_cache_lock:
        if now < _hkex_stock_cache[0]:
            return _hkex_stock_cache[1]
    payload = json.loads(
        _read_bytes(
            "https://www1.hkexnews.hk/ncms/script/eds/activestock_sehk_e.json",
            timeout_seconds=10,
        )
    )
    mapping = {
        str(item.get("c") or "").zfill(5): (
            int(item.get("i") or 0),
            str(item.get("n") or ""),
        )
        for item in payload or []
        if item.get("c") and item.get("i")
    }
    with _reference_cache_lock:
        _hkex_stock_cache = (now + 86_400, mapping)
    return mapping


def _fetch_hkex_announcements(
    symbol: str,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    digits = "".join(character for character in symbol if character.isdigit())
    code = digits.zfill(5)
    company = _hkex_company_map().get(code)
    if company is None:
        return []
    stock_id, stock_name = company
    url = (
        "https://www1.hkexnews.hk/search/titlesearch.xhtml?"
        + urlencode({"category": 0, "market": "SEHK", "stockId": stock_id})
    )
    parser = _HKEXTitleParser()
    parser.feed(
        _read_bytes(url, timeout_seconds=timeout_seconds).decode(
            "utf-8", errors="replace"
        )
    )
    items: list[NewsArticle] = []
    for row in parser.rows:
        published = re.sub(r"^\s*Release Time:\s*", "", row["published"]).strip()
        title = _plain_text(row["title"], 220)
        headline = _plain_text(row["headline"], 180)
        item = _article(
            title=title,
            publisher="HKEXnews",
            published=published,
            url=row["url"],
            provider="hkexnews",
            provider_label="HKEXnews 官方公告",
            summary=f"{stock_name} · {headline}" if headline else stock_name,
            offset_hours=8,
            provider_url="https://www.hkexnews.hk/",
            source_kind="filing",
            credibility="交易所官方披露",
        )
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def _error_message(error: BaseException) -> str:
    text = re.sub(r"\s+", " ", str(error)).strip()
    return (text or error.__class__.__name__)[:180]


def _relevance_terms(symbol: str, company_name: str) -> tuple[str, ...]:
    stop_words = {
        "company",
        "corporation",
        "corp",
        "group",
        "holdings",
        "holding",
        "limited",
        "ltd",
        "stock",
        "shares",
        "motor",
        "基金",
        "股份",
        "公司",
        "集团",
        "控股",
    }
    base_symbol = re.sub(r"[^A-Za-z0-9]", "", symbol.split(".", 1)[0]).casefold()
    raw_terms = re.findall(
        r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}",
        company_name.casefold(),
    )
    terms = {
        term
        for term in [base_symbol, *raw_terms]
        if len(term) >= 3 and term not in stop_words
    }
    company_phrase = re.sub(r"\s+", " ", company_name.casefold()).strip()
    if len(company_phrase) >= 4:
        terms.add(company_phrase)
    return tuple(sorted(terms, key=lambda term: (-len(term), term)))


def _deduplicate(
    items: list[NewsArticle],
    limit: int,
    *,
    relevance_terms: tuple[str, ...] = (),
) -> tuple[NewsArticle, ...]:
    def timestamp(item: NewsArticle) -> float:
        if not item.published_at:
            return 0.0
        try:
            return datetime.fromisoformat(item.published_at).timestamp()
        except ValueError:
            return 0.0

    def relevance(item: NewsArticle) -> int:
        haystack = f"{item.title} {item.summary}".casefold()
        source_score = 2 if item.source_kind in {"filing", "exchange"} else 0
        term_score = sum(3 for term in relevance_terms if term in haystack)
        return source_score + term_score

    ordered = sorted(
        items,
        key=lambda item: (relevance(item), timestamp(item)),
        reverse=True,
    )
    selected: list[NewsArticle] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in ordered:
        title_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", item.title.casefold())
        url_key = urlunsplit(
            (
                urlsplit(item.url).scheme,
                urlsplit(item.url).netloc.casefold(),
                urlsplit(item.url).path.rstrip("/"),
                "",
                "",
            )
        )
        if url_key in seen_urls or (title_key and title_key in seen_titles):
            continue
        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return tuple(selected)


_SOURCE_ALIAS_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sec", "edgar"), "sec-edgar"),
    (("hkex",), "hkex"),
    (("cninfo", "巨潮"), "cninfo"),
    (("eastmoney", "东方财富"), "eastmoney"),
    (("reuters", "路透"), "reuters"),
    (("bloomberg", "彭博"), "bloomberg"),
    (("apnews", "associatedpress", "美联社"), "associated-press"),
    (("motleyfool", "foolcom"), "motley-fool"),
    (("yahoofinance",), "yahoo-finance"),
    (("nasdaq",), "nasdaq"),
    (("stcn", "证券时报"), "stcn"),
    (("cnstock", "上海证券报"), "cnstock"),
)
_TWO_LEVEL_SUFFIXES = {
    "co.uk",
    "com.au",
    "com.cn",
    "com.hk",
    "com.sg",
    "co.jp",
    "net.cn",
    "org.cn",
}
_EVENT_STOP_WORDS = {
    "about",
    "after",
    "amid",
    "announces",
    "company",
    "corporation",
    "from",
    "into",
    "latest",
    "limited",
    "market",
    "news",
    "shares",
    "stock",
    "that",
    "this",
    "update",
    "with",
}


def _official_source(item: NewsArticle) -> bool:
    return item.source_kind in {"filing", "exchange"} or any(
        word in item.credibility for word in ("官方", "法定", "监管")
    )


def _domain_source_id(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    parts = [part for part in hostname.split(".") if part]
    if len(parts) < 2:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", hostname)
    suffix = ".".join(parts[-2:])
    root_index = -3 if suffix in _TWO_LEVEL_SUFFIXES and len(parts) >= 3 else -2
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", parts[root_index])


def _source_identity(item: NewsArticle) -> str:
    publisher = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+", "", item.publisher.casefold()
    )
    domain_id = _domain_source_id(item.url)
    combined = f"{publisher} {domain_id}"
    for aliases, canonical in _SOURCE_ALIAS_RULES:
        if any(alias in combined for alias in aliases):
            return canonical
    if "." in item.publisher and domain_id:
        return domain_id
    generic = {
        "",
        "unknown",
        "sourceunknown",
        "来源未知",
        "财经媒体",
        "testwire",
    }
    if publisher not in generic:
        for suffix in ("inc", "ltd", "limited", "company", "media", "newswire"):
            publisher = publisher.removesuffix(suffix)
        if publisher:
            return publisher
    return domain_id or re.sub(r"[^a-z0-9]+", "", item.provider.casefold())


def _event_category(item: NewsArticle) -> str:
    text = f"{item.title} {item.summary}".casefold()
    categories = (
        ("results", ("业绩", "财报", "营收", "净利润", "earnings", "revenue", "quarterly results")),
        ("guidance", ("预告", "盈利预测", "guidance", "forecast", "outlook")),
        ("capital-return", ("分红", "派息", "回购", "dividend", "buyback", "repurchase")),
        ("financing", ("定增", "配股", "融资", "可转债", "offering", "financing", "convertible notes")),
        ("transaction", ("收购", "并购", "出售", "入股", "acquisition", "merger", "stake sale")),
        ("product-contract", ("中标", "合同", "订单", "获批", "发布", "launch", "contract", "approval", "order")),
        ("regulatory-risk", ("调查", "处罚", "诉讼", "召回", "违约", "probe", "lawsuit", "recall", "sanction")),
        ("management", ("任命", "辞任", "董事长", "总经理", "chief executive", "chairman", "resigns")),
        ("analyst-rating", ("评级", "目标价", "上调", "下调", "upgrade", "downgrade", "price target")),
    )
    for category, terms in categories:
        if any(term in text for term in terms):
            return category
    return "company-update"


def _event_features(
    item: NewsArticle,
    relevance_terms: tuple[str, ...],
) -> tuple[str, set[str], set[str]]:
    text = item.title.casefold()
    for term in relevance_terms:
        text = text.replace(term.casefold(), " ")
    text = re.sub(r"\s+[|\-–:]\s+[^|\-–:]{2,40}$", " ", text)
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    tokens: set[str] = set()
    numbers: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
        if raw.isdigit():
            if len(raw) >= 2:
                numbers.add(raw)
                tokens.add(raw)
            continue
        if re.fullmatch(r"[a-z0-9]+", raw):
            if len(raw) >= 3 and raw not in _EVENT_STOP_WORDS:
                tokens.add(raw)
            continue
        cleaned = raw
        for word in ("公告", "公司", "股份", "集团", "关于", "发布"):
            cleaned = cleaned.replace(word, "")
        if len(cleaned) == 2:
            tokens.add(cleaned)
        elif len(cleaned) > 2:
            tokens.update(cleaned[index : index + 2] for index in range(len(cleaned) - 1))
    return normalized, tokens, numbers


def _article_datetime(item: NewsArticle) -> datetime | None:
    if not item.published_at:
        return None
    try:
        value = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _event_similarity(
    left: NewsArticle,
    right: NewsArticle,
    relevance_terms: tuple[str, ...],
) -> float:
    left_time = _article_datetime(left)
    right_time = _article_datetime(right)
    if left_time and right_time and abs((left_time - right_time).total_seconds()) > 96 * 3_600:
        return 0.0
    left_category = _event_category(left)
    right_category = _event_category(right)
    left_normalized, left_tokens, left_numbers = _event_features(left, relevance_terms)
    right_normalized, right_tokens, right_numbers = _event_features(right, relevance_terms)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union if union else 0.0
    overlap = intersection / min(len(left_tokens), len(right_tokens)) if left_tokens and right_tokens else 0.0
    category_match = left_category == right_category
    if not category_match and left_category != "company-update" and right_category != "company-update":
        return sequence_score if sequence_score >= 0.92 else 0.0
    if left_numbers and right_numbers and not (left_numbers & right_numbers) and sequence_score < 0.88:
        return 0.0
    score = max(sequence_score, jaccard * 0.9 + overlap * 0.1)
    if sequence_score >= 0.82:
        return score
    if intersection >= 3 and jaccard >= 0.44 and overlap >= 0.62:
        return score
    if category_match and intersection >= 4 and overlap >= 0.72:
        return score
    return 0.0


def _aggregate_events(
    items: list[NewsArticle],
    limit: int,
    *,
    relevance_terms: tuple[str, ...] = (),
) -> tuple[tuple[NewsArticle, ...], NewsVerification]:
    def rank(item: NewsArticle) -> tuple[int, int, float]:
        haystack = f"{item.title} {item.summary}".casefold()
        relevance = sum(3 for term in relevance_terms if term in haystack)
        published = _article_datetime(item)
        return (
            relevance,
            1 if _official_source(item) else 0,
            published.timestamp() if published else 0.0,
        )

    clusters: list[list[NewsArticle]] = []
    for item in sorted(items, key=rank, reverse=True):
        best_cluster: list[NewsArticle] | None = None
        best_score = 0.0
        for cluster in clusters:
            similarity = max(
                _event_similarity(item, existing, relevance_terms)
                for existing in cluster
            )
            if similarity > best_score:
                best_score = similarity
                best_cluster = cluster
        if best_cluster is not None and best_score > 0:
            best_cluster.append(item)
        else:
            clusters.append([item])

    events: list[NewsArticle] = []
    for cluster in clusters:
        representative = max(
            cluster,
            key=lambda item: (
                1 if _official_source(item) else 0,
                rank(item)[0],
                len(item.summary),
                rank(item)[2],
            ),
        )
        source_articles: dict[str, NewsArticle] = {}
        for item in cluster:
            source_id = _source_identity(item)
            current = source_articles.get(source_id)
            if current is None or (
                _official_source(item), len(item.summary), rank(item)[2]
            ) > (
                _official_source(current), len(current.summary), rank(current)[2]
            ):
                source_articles[source_id] = item
        references = tuple(
            NewsSourceReference(
                source_id=source_id,
                title=item.title,
                publisher=item.publisher,
                published_at=item.published_at,
                url=item.url,
                provider=item.provider,
                provider_label=item.provider_label,
                source_kind=item.source_kind,
                credibility=item.credibility,
                official=_official_source(item),
            )
            for source_id, item in sorted(
                source_articles.items(),
                key=lambda pair: (
                    1 if _official_source(pair[1]) else 0,
                    rank(pair[1])[2],
                ),
                reverse=True,
            )
        )
        source_count = len(references)
        provider_count = len({item.provider for item in cluster})
        official_count = sum(reference.official for reference in references)
        if source_count >= VERIFICATION_SOURCE_THRESHOLD:
            status = "five-source"
        elif official_count:
            status = "official-primary"
        elif source_count >= 2:
            status = "corroborated"
        else:
            status = "single-source"
        verification_score = round(
            100
            * (
                0.75 * min(source_count / VERIFICATION_SOURCE_THRESHOLD, 1.0)
                + 0.10 * min(provider_count / 3.0, 1.0)
                + 0.15 * min(official_count, 1)
            ),
            1,
        )
        dates = sorted(
            date for date in (_article_datetime(item) for item in cluster) if date
        )
        richest_summary = max(
            (item.summary for item in cluster),
            key=lambda value: len(value),
            default=representative.summary,
        )
        event_key = sha256(
            (
                f"{_event_category(representative)}|"
                f"{_event_features(representative, relevance_terms)[0]}|"
                f"{dates[0].date().isoformat() if dates else ''}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        events.append(
            replace(
                representative,
                article_id=event_key,
                summary=richest_summary,
                event_id=event_key,
                event_category=_event_category(representative),
                verification_status=status,
                verification_count=source_count,
                provider_count=provider_count,
                official_source_count=official_count,
                verification_score=verification_score,
                first_reported_at=(dates[0].isoformat(timespec="seconds") if dates else ""),
                last_reported_at=(dates[-1].isoformat(timespec="seconds") if dates else ""),
                corroborating_sources=references[:20],
            )
        )

    status_rank = {
        "five-source": 3,
        "official-primary": 2,
        "corroborated": 1,
        "single-source": 0,
    }
    ordered_events = sorted(
        events,
        key=lambda item: (
            rank(item)[0],
            status_rank[item.verification_status],
            item.verification_score,
            rank(item)[2],
        ),
        reverse=True,
    )
    selected: list[NewsArticle] = []
    deferred: list[NewsArticle] = []
    source_event_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    filing_count = 0
    source_cap = max(2, math.ceil(limit / 3))
    category_cap = max(3, math.ceil(limit / 2))
    filing_cap = max(2, math.ceil(limit / 3))
    for event in ordered_events:
        source_id = _source_identity(event)
        over_cap = (
            source_event_counts.get(source_id, 0) >= source_cap
            or category_counts.get(event.event_category, 0) >= category_cap
            or (event.source_kind == "filing" and filing_count >= filing_cap)
        )
        if over_cap:
            deferred.append(event)
            continue
        selected.append(event)
        source_event_counts[source_id] = source_event_counts.get(source_id, 0) + 1
        category_counts[event.event_category] = category_counts.get(event.event_category, 0) + 1
        filing_count += event.source_kind == "filing"
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])
    source_ids = {
        reference.source_id
        for event in selected
        for reference in event.corroborating_sources
    }
    verified_count = sum(
        event.verification_status == "five-source" for event in selected
    )
    verification = NewsVerification(
        raw_article_count=len(items),
        event_count=len(selected),
        five_source_verified_count=verified_count,
        official_primary_count=sum(
            event.official_source_count > 0 for event in selected
        ),
        corroborated_count=sum(event.verification_count >= 2 for event in selected),
        independent_source_count=len(source_ids),
        verified_event_ratio=round(verified_count / len(selected), 6) if selected else 0.0,
    )
    return tuple(selected), verification


def fetch_online_news(
    market: str,
    symbol: str,
    *,
    company_name: str = "",
    limit: int = 12,
    timeout_seconds: float = 12,
) -> NewsFeed:
    normalized_market = market.strip().lower()
    normalized_symbol = symbol.strip().upper()
    bounded_limit = max(1, min(int(limit), 30))
    provider_limit = min(30, max(12, bounded_limit * 2))
    per_request_timeout = max(2.0, min(8.0, timeout_seconds - 0.5))
    company = _plain_text(company_name, 80) or normalized_symbol
    provider_specs: tuple[
        tuple[str, str, str, str, str, Callable[[], list[NewsArticle]]], ...
    ]
    source_portals: tuple[dict[str, str], ...]
    if normalized_market == "a-share":
        query = f"{company} {normalized_symbol} 股票"
        provider_specs = (
            (
                "eastmoney",
                "东方财富（AKShare）",
                "https://finance.eastmoney.com/",
                "media",
                "主流财经门户",
                lambda: _fetch_eastmoney_news(normalized_symbol, provider_limit),
            ),
            (
                "cninfo",
                "巨潮资讯公告",
                "https://www.cninfo.com.cn/",
                "filing",
                "法定信息披露平台",
                lambda: _fetch_cninfo_announcements(
                    normalized_symbol, provider_limit
                ),
            ),
            (
                "gdelt",
                "GDELT 全球新闻索引",
                "https://www.gdeltproject.org/",
                "aggregator",
                "跨媒体原始报道索引",
                lambda: _fetch_gdelt_news(
                    normalized_symbol, company, provider_limit, per_request_timeout
                ),
            ),
        )
        source_portals = (
            {"label": "巨潮资讯", "url": "https://www.cninfo.com.cn/", "kind": "法定披露"},
            {"label": "上交所", "url": "https://www.sse.com.cn/disclosure/listedinfo/announcement/", "kind": "交易所披露"},
            {"label": "深交所", "url": "https://www.szse.cn/disclosure/notice/company/index.html", "kind": "交易所披露"},
            {"label": "东方财富", "url": "https://finance.eastmoney.com/", "kind": "财经媒体"},
            {"label": "GDELT", "url": "https://www.gdeltproject.org/", "kind": "跨媒体索引"},
        )
    elif normalized_market == "nasdaq":
        query = f"{company} {normalized_symbol} stock"
        provider_specs = (
            (
                "yahoo-finance",
                "Yahoo Finance",
                "https://finance.yahoo.com/",
                "aggregator",
                "国际财经门户",
                lambda: _fetch_yahoo_news(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "nasdaq-rss",
                "Nasdaq 官方 RSS",
                "https://www.nasdaq.com/nasdaq-rss-feeds",
                "exchange-media",
                "交易所新闻频道",
                lambda: _fetch_nasdaq_news(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "sec-edgar",
                "SEC EDGAR 官方文件",
                "https://www.sec.gov/edgar/search/",
                "filing",
                "监管机构官方",
                lambda: _fetch_sec_filings(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "gdelt",
                "GDELT 全球新闻索引",
                "https://www.gdeltproject.org/",
                "aggregator",
                "跨媒体原始报道索引",
                lambda: _fetch_gdelt_news(
                    normalized_symbol, company, provider_limit, per_request_timeout
                ),
            ),
        )
        source_portals = (
            {"label": "SEC EDGAR", "url": "https://www.sec.gov/edgar/search/", "kind": "监管披露"},
            {"label": "Nasdaq", "url": "https://www.nasdaq.com/nasdaq-rss-feeds", "kind": "交易所"},
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "财经媒体"},
            {"label": "GDELT", "url": "https://www.gdeltproject.org/", "kind": "跨媒体索引"},
        )
    elif normalized_market == "hk":
        query = f"{company} {normalized_symbol} Hong Kong stock"
        provider_specs = (
            (
                "hkexnews",
                "HKEXnews 官方公告",
                "https://www.hkexnews.hk/",
                "filing",
                "交易所官方披露",
                lambda: _fetch_hkex_announcements(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "yahoo-finance",
                "Yahoo Finance",
                "https://finance.yahoo.com/",
                "aggregator",
                "国际财经门户",
                lambda: _fetch_yahoo_news(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "gdelt",
                "GDELT 全球新闻索引",
                "https://www.gdeltproject.org/",
                "aggregator",
                "跨媒体原始报道索引",
                lambda: _fetch_gdelt_news(
                    normalized_symbol, company, provider_limit, per_request_timeout
                ),
            ),
        )
        source_portals = (
            {"label": "HKEXnews", "url": "https://www.hkexnews.hk/", "kind": "交易所披露"},
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "财经媒体"},
            {"label": "GDELT", "url": "https://www.gdeltproject.org/", "kind": "跨媒体索引"},
        )
    elif normalized_market == "global":
        query = f"{company} {normalized_symbol} stock ETF"
        provider_specs = (
            (
                "yahoo-finance",
                "Yahoo Finance",
                "https://finance.yahoo.com/",
                "aggregator",
                "国际财经门户",
                lambda: _fetch_yahoo_news(
                    normalized_symbol, provider_limit, per_request_timeout
                ),
            ),
            (
                "gdelt",
                "GDELT 全球新闻索引",
                "https://www.gdeltproject.org/",
                "aggregator",
                "跨媒体原始报道索引",
                lambda: _fetch_gdelt_news(
                    normalized_symbol, company, provider_limit, per_request_timeout
                ),
            ),
        )
        source_portals = (
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "国际财经媒体"},
            {"label": "Microsoft Finance", "url": "https://www.msn.com/en-us/money", "kind": "行情与公司资料"},
            {"label": "GDELT", "url": "https://www.gdeltproject.org/", "kind": "跨媒体索引"},
        )
    else:
        return NewsFeed(
            market=normalized_market,
            symbol=normalized_symbol,
            query=company,
            items=(),
            providers=(),
            warnings=("当前市场不提供在线新闻。",),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_portals=(),
        )

    results: queue.Queue[tuple[str, list[NewsArticle] | None, BaseException | None]] = (
        queue.Queue()
    )

    def run_provider(
        provider: str,
        loader: Callable[[], list[NewsArticle]],
    ) -> None:
        try:
            results.put((provider, loader(), None))
        except Exception as error:
            results.put((provider, None, error))

    workers: dict[str, threading.Thread] = {}
    for provider, _label, _url, _kind, _credibility, loader in provider_specs:
        worker = threading.Thread(
            target=run_provider,
            args=(provider, loader),
            daemon=True,
            name=f"news-{provider}",
        )
        workers[provider] = worker
        worker.start()

    deadline = time.monotonic() + max(2.0, timeout_seconds)
    completed: dict[str, tuple[list[NewsArticle] | None, BaseException | None]] = {}
    while len(completed) < len(provider_specs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            provider, items, error = results.get(timeout=remaining)
        except queue.Empty:
            break
        completed[provider] = (items, error)

    statuses: list[NewsProviderStatus] = []
    collected: list[NewsArticle] = []
    warnings: list[str] = []
    for provider, label, source_url, source_kind, credibility, _loader in provider_specs:
        if provider not in completed:
            message = f"{label} 获取超时"
            statuses.append(
                NewsProviderStatus(
                    provider,
                    label,
                    "timeout",
                    0,
                    message,
                    source_url,
                    source_kind,
                    credibility,
                )
            )
            warnings.append(message)
            continue
        items, error = completed[provider]
        if error is not None:
            message = _error_message(error)
            statuses.append(
                NewsProviderStatus(
                    provider,
                    label,
                    "error",
                    0,
                    message,
                    source_url,
                    source_kind,
                    credibility,
                )
            )
            warnings.append(f"{label} 暂不可用：{message}")
            continue
        usable = items or []
        collected.extend(usable)
        statuses.append(
            NewsProviderStatus(
                provider,
                label,
                "ok" if usable else "empty",
                len(usable),
                "" if usable else "未返回相关新闻",
                source_url,
                source_kind,
                credibility,
            )
        )

    aggregated, verification = _aggregate_events(
        collected,
        bounded_limit,
        relevance_terms=_relevance_terms(normalized_symbol, company),
    )
    if not aggregated and not warnings:
        warnings.append("在线新闻源暂未返回该标的的相关新闻。")
    return NewsFeed(
        market=normalized_market,
        symbol=normalized_symbol,
        query=query,
        items=aggregated,
        providers=tuple(statuses),
        warnings=tuple(warnings),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_portals=source_portals,
        verification=verification,
    )
