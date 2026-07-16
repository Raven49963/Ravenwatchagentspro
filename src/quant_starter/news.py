from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36 RavenWatchAgentsPro/1.6"
)
SEC_USER_AGENT = os.environ.get(
    "RAVENWATCHAGENTSPRO_SEC_USER_AGENT",
    "RavenWatchAgentsPro/1.6 quant-research contact@example.com",
).strip()


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

    def to_dict(self) -> dict[str, str]:
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
class NewsFeed:
    market: str
    symbol: str
    query: str
    items: tuple[NewsArticle, ...]
    providers: tuple[NewsProviderStatus, ...]
    warnings: tuple[str, ...]
    fetched_at: str
    source_portals: tuple[dict[str, str], ...] = ()

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
        source_kind="exchange",
        credibility="交易所官方",
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
        )
        source_portals = (
            {"label": "巨潮资讯", "url": "https://www.cninfo.com.cn/", "kind": "法定披露"},
            {"label": "东方财富", "url": "https://finance.eastmoney.com/", "kind": "财经媒体"},
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
                "exchange",
                "交易所官方",
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
        )
        source_portals = (
            {"label": "SEC EDGAR", "url": "https://www.sec.gov/edgar/search/", "kind": "监管披露"},
            {"label": "Nasdaq", "url": "https://www.nasdaq.com/nasdaq-rss-feeds", "kind": "交易所"},
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "财经媒体"},
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
        )
        source_portals = (
            {"label": "HKEXnews", "url": "https://www.hkexnews.hk/", "kind": "交易所披露"},
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "财经媒体"},
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
        )
        source_portals = (
            {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "kind": "国际财经媒体"},
            {"label": "Microsoft Finance", "url": "https://www.msn.com/en-us/money", "kind": "行情与公司资料"},
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

    deduplicated = _deduplicate(
        collected,
        bounded_limit,
        relevance_terms=_relevance_terms(normalized_symbol, company),
    )
    if not deduplicated and not warnings:
        warnings.append("在线新闻源暂未返回该标的的相关新闻。")
    return NewsFeed(
        market=normalized_market,
        symbol=normalized_symbol,
        query=query,
        items=deduplicated,
        providers=tuple(statuses),
        warnings=tuple(warnings),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_portals=source_portals,
    )
