from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from quant_starter.data import (
    DemoMarketConfig,
    fetch_a_share_ohlcv,
    fetch_global_ohlcv,
    fetch_hk_ohlcv,
    fetch_nasdaq_ohlcv,
    generate_demo_ohlcv,
    load_ohlcv_csv,
    normalize_a_share_symbol,
    normalize_hk_symbol,
    provider_display_name,
    summarize_error,
)
from quant_starter.global_market import fetch_msn_quote, resolve_msn_instrument
from quant_starter.news import SEC_USER_AGENT, _read_bytes, _sec_company_map


@dataclass
class ResearchContext:
    symbol: str
    market: str
    analysis_date: str
    bars: pd.DataFrame
    technical: dict[str, float | str | None]
    fundamentals: dict[str, Any] = field(default_factory=dict)
    news: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "analysis_date": self.analysis_date,
            "technical_snapshot": self.technical,
            "fundamentals": self.fundamentals,
            "news": self.news[:12],
            "data_warnings": self.warnings,
        }


@dataclass(frozen=True)
class FundamentalProviderStatus:
    provider: str
    label: str
    status: str
    field_count: int
    message: str = ""
    source_url: str = ""
    source_kind: str = "market-data"
    credibility: str = "公开数据源"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FundamentalSnapshot:
    market: str
    symbol: str
    fields: dict[str, Any]
    field_sources: dict[str, str]
    providers: tuple[FundamentalProviderStatus, ...]
    warnings: tuple[str, ...]
    fetched_at: str
    field_evidence: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "fields": self.fields,
            "field_sources": self.field_sources,
            "providers": [provider.to_dict() for provider in self.providers],
            "warnings": list(self.warnings),
            "fetched_at": self.fetched_at,
            "field_evidence": {
                key: list(entries) for key, entries in self.field_evidence.items()
            },
            "quality": self.quality,
        }


def _run_with_timeout(function, timeout_seconds: int):
    """Run optional enrichment in a daemon thread so it cannot stall research."""

    results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            results.put(("ok", function()))
        except Exception as exc:
            results.put(("error", exc))

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"数据补充步骤超过 {timeout_seconds} 秒")
    status, payload = results.get_nowait()
    if status == "error":
        if isinstance(payload, BaseException):
            raise payload
        raise RuntimeError(f"数据补充步骤失败：{payload}")
    return payload


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _latest(series: pd.Series) -> float | None:
    usable = series.dropna()
    return _number(usable.iloc[-1]) if not usable.empty else None


def calculate_technical_snapshot(bars: pd.DataFrame) -> dict[str, float | str | None]:
    """Build deterministic, auditable technical evidence for every agent."""

    close = bars["Close"].astype(float)
    volume = bars["Volume"].astype(float)
    returns = close.pct_change()
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi14 = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi14 = rsi14.where(loss != 0, 100.0).where(gain != 0, 0.0)
    rsi14 = rsi14.where(~((gain == 0) & (loss == 0)), 50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - macd_signal

    running_peak = close.cummax()
    drawdown = close / running_peak - 1.0
    volume20 = volume.rolling(20).mean()

    return {
        "last_bar_date": str(bars.index[-1].date()),
        "close": _latest(close),
        "sma5": _latest(sma5),
        "sma20": _latest(sma20),
        "sma60": _latest(sma60),
        "rsi14": _latest(rsi14),
        "macd": _latest(macd),
        "macd_signal": _latest(macd_signal),
        "macd_histogram": _latest(macd_histogram),
        "return_5d": _number(close.pct_change(5).iloc[-1]),
        "return_20d": _number(close.pct_change(20).iloc[-1]),
        "return_60d": _number(close.pct_change(60).iloc[-1]),
        "annualized_volatility": _number(returns.std(ddof=0) * np.sqrt(252)),
        "max_drawdown": _number(drawdown.min()),
        "volume_ratio_20d": _number(volume.iloc[-1] / volume20.iloc[-1])
        if _number(volume20.iloc[-1]) not in {None, 0.0}
        else None,
        "rows": float(len(bars)),
    }


def _nasdaq_details(symbol: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    yfinance = __import__("yfinance")
    ticker = yfinance.Ticker(symbol)
    raw_info = ticker.get_info() or {}
    wanted = (
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "profitMargins",
        "revenueGrowth",
        "debtToEquity",
        "returnOnEquity",
    )
    fundamentals = {key: raw_info.get(key) for key in wanted if raw_info.get(key) is not None}

    news = []
    for item in (ticker.news or [])[:12]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
        canonical = content.get("canonicalUrl", {})
        url = canonical.get("url", "") if isinstance(canonical, dict) else ""
        if title:
            news.append({"title": str(title), "publisher": str(publisher), "url": str(url)})
    return fundamentals, news


def _yfinance_fundamentals(symbol: str) -> dict[str, Any]:
    yfinance = __import__("yfinance")
    ticker = yfinance.Ticker(symbol)
    raw_info = ticker.get_info() or {}
    wanted = (
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "revenueGrowth",
        "earningsGrowth",
        "debtToEquity",
        "returnOnEquity",
        "returnOnAssets",
        "currentRatio",
        "quickRatio",
        "freeCashflow",
        "operatingCashflow",
        "totalRevenue",
    )
    fields = {key: raw_info.get(key) for key in wanted if raw_info.get(key) is not None}
    revenue = _number(fields.get("totalRevenue"))
    free_cash_flow = _number(fields.get("freeCashflow"))
    operating_cash_flow = _number(fields.get("operatingCashflow"))
    if revenue not in {None, 0.0}:
        if free_cash_flow is not None:
            fields["freeCashFlowMargin"] = free_cash_flow / revenue
        if operating_cash_flow is not None:
            fields["operatingCashFlowMargin"] = operating_cash_flow / revenue
    fields["_source_url"] = f"https://finance.yahoo.com/quote/{symbol}"
    return fields


def _a_share_details(symbol: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    akshare = __import__("akshare")
    fundamentals: dict[str, Any] = {}
    info = akshare.stock_individual_info_em(symbol=symbol)
    if not info.empty and {"item", "value"}.issubset(info.columns):
        fundamentals = {
            str(row["item"]): row["value"] for _, row in info.iterrows()
        }

    news = []
    raw_news = akshare.stock_news_em(symbol=symbol)
    if not raw_news.empty and "新闻标题" in raw_news.columns:
        for _, row in raw_news.head(12).iterrows():
            news.append(
                {
                    "title": str(row.get("新闻标题", "")),
                    "publisher": str(row.get("文章来源", "")),
                    "published": str(row.get("发布时间", "")),
                    "url": str(row.get("新闻链接", "")),
                }
            )
    return fundamentals, news


def _msn_details(
    market: str,
    symbol: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    instrument = resolve_msn_instrument(symbol, market)
    quote = fetch_msn_quote(instrument)
    wanted = (
        "displayName",
        "exchangeName",
        "country",
        "securityType",
        "marketCap",
        "marketCapCurrency",
        "peRatio",
        "yieldPercent",
        "price52wHigh",
        "price52wLow",
        "averageVolume",
        "return1Month",
        "return3Month",
        "return1Year",
    )
    fundamentals = {key: quote.get(key) for key in wanted if quote.get(key) is not None}
    fundamentals.update(
        instrument_id=instrument.instrument_id,
        exchange=instrument.exchange,
        asset_type=instrument.asset_type,
        source="Microsoft Finance",
        source_url=instrument.source_url,
    )
    return fundamentals, []


def _msn_fundamentals(market: str, symbol: str) -> dict[str, Any]:
    fields, _ = _msn_details(market, symbol)
    source_url = str(fields.pop("source_url", ""))
    fields.pop("source", None)
    fields["_source_url"] = source_url
    return fields


def _a_share_financials(symbol: str) -> dict[str, Any]:
    akshare = __import__("akshare")
    start_year = str(max(2000, datetime.now().year - 2))
    frame = akshare.stock_financial_analysis_indicator(
        symbol=symbol,
        start_year=start_year,
    )
    if frame.empty or "日期" not in frame.columns:
        return {}
    dates = pd.to_datetime(frame["日期"], errors="coerce")
    if dates.notna().sum() == 0:
        return {}
    row = frame.loc[dates.idxmax()]

    def percentage(column: str) -> float | None:
        value = _number(row.get(column))
        return value / 100 if value is not None else None

    fields = {
        "reportDate": dates.max().date().isoformat(),
        "profitMargins": percentage("销售净利率(%)"),
        "revenueGrowth": percentage("主营业务收入增长率(%)"),
        "earningsGrowth": percentage("净利润增长率(%)"),
        "returnOnEquity": percentage("净资产收益率(%)"),
        "debtToEquity": _number(row.get("负债与所有者权益比率(%)")),
        "debtRatio": percentage("资产负债率(%)"),
        "currentRatio": _number(row.get("流动比率")),
        "quickRatio": _number(row.get("速动比率")),
        "earningsPerShare": _number(row.get("摊薄每股收益(元)")),
        "bookValuePerShare": _number(row.get("每股净资产_调整后(元)")),
        "totalAssets": _number(row.get("总资产(元)")),
        "_source_url": f"https://finance.sina.com.cn/realstock/company/{'sh' if symbol.startswith(('5', '6', '9')) else 'sz'}{symbol}/nc.shtml",
    }
    return {key: value for key, value in fields.items() if value is not None}


def _eastmoney_financial_analysis(market: str, symbol: str) -> dict[str, Any]:
    akshare = __import__("akshare")
    if market == "a-share":
        suffix = "SH" if symbol.startswith(("5", "6", "9")) else "BJ" if symbol.startswith(("4", "8")) else "SZ"
        frame = akshare.stock_financial_analysis_indicator_em(
            symbol=f"{symbol}.{suffix}",
            indicator="按报告期",
        )
    elif market == "hk":
        digits = normalize_hk_symbol(symbol).removesuffix(".HK").zfill(5)
        frame = akshare.stock_financial_hk_analysis_indicator_em(
            symbol=digits,
            indicator="报告期",
        )
    elif market == "nasdaq":
        frame = akshare.stock_financial_us_analysis_indicator_em(
            symbol=symbol,
            indicator="年报",
        )
    else:
        return {}
    if frame is None or frame.empty or "REPORT_DATE" not in frame.columns:
        return {}
    dates = pd.to_datetime(frame["REPORT_DATE"], errors="coerce")
    if not dates.notna().any():
        return {}
    row = frame.loc[dates.idxmax()]

    def ratio(column: str) -> float | None:
        value = _number(row.get(column))
        return value / 100 if value is not None else None

    common = {
        "reportDate": dates.max().date().isoformat(),
        "longName": row.get("SECURITY_NAME_ABBR"),
    }
    if market == "a-share":
        common.update(
            revenueGrowth=ratio("TOTALOPERATEREVETZ"),
            earningsGrowth=ratio("PARENTNETPROFITTZ"),
            profitMargins=ratio("XSJLL"),
            grossMargins=ratio("XSMLL"),
            returnOnEquity=ratio("ROEJQ"),
            returnOnAssets=ratio("ZZCJLL"),
            operatingCashFlowMargin=ratio("JYXJLYYSR"),
            debtRatio=ratio("ZCFZL"),
            debtToEquityRatio=ratio("CQBL"),
            currentRatio=_number(row.get("LD")),
            quickRatio=_number(row.get("SD")),
            annualRevenue=_number(row.get("TOTALOPERATEREVE")),
            annualNetIncome=_number(row.get("PARENTNETPROFIT")),
            operatingCashFlowPerShare=_number(row.get("MGJYXJJE")),
            _source_url=f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={suffix}{symbol}#/cwfx",
        )
    elif market == "hk":
        common.update(
            revenueGrowth=ratio("OPERATE_INCOME_YOY"),
            earningsGrowth=ratio("HOLDER_PROFIT_YOY"),
            profitMargins=ratio("NET_PROFIT_RATIO"),
            grossMargins=ratio("GROSS_PROFIT_RATIO"),
            returnOnEquity=ratio("ROE_AVG"),
            returnOnAssets=ratio("ROA"),
            operatingCashFlowMargin=ratio("OCF_SALES"),
            debtRatio=ratio("DEBT_ASSET_RATIO"),
            currentRatio=_number(row.get("CURRENT_RATIO")),
            annualRevenue=_number(row.get("OPERATE_INCOME")),
            annualNetIncome=_number(row.get("HOLDER_PROFIT")),
            operatingCashFlowPerShare=_number(row.get("PER_NETCASH_OPERATE")),
            _source_url=f"https://emweb.securities.eastmoney.com/PC_HKF10/NewFinancialAnalysis/index?type=web&code={digits}",
        )
    else:
        common.update(
            revenueGrowth=ratio("OPERATE_INCOME_YOY"),
            earningsGrowth=ratio("PARENT_HOLDER_NETPROFIT_YOY"),
            profitMargins=ratio("NET_PROFIT_RATIO"),
            grossMargins=ratio("GROSS_PROFIT_RATIO"),
            returnOnEquity=ratio("ROE_AVG"),
            returnOnAssets=ratio("ROA"),
            debtRatio=ratio("DEBT_ASSET_RATIO"),
            currentRatio=_number(row.get("CURRENT_RATIO")),
            quickRatio=_number(row.get("SPEED_RATIO")),
            annualRevenue=_number(row.get("OPERATE_INCOME")),
            annualNetIncome=_number(row.get("PARENT_HOLDER_NETPROFIT")),
            _source_url=f"https://emweb.eastmoney.com/PC_USF10/pages/index.html?code={symbol}#/cwfx/zyzb",
        )
    return {
        key: value
        for key, value in common.items()
        if value is not None and not (isinstance(value, float) and not np.isfinite(value))
    }


def _sec_company_fundamentals(symbol: str) -> dict[str, Any]:
    lookup = symbol.replace("-", ".").upper()
    company = _sec_company_map().get(lookup) or _sec_company_map().get(symbol.upper())
    if company is None:
        return {}
    cik, company_name = company
    payload = json.loads(
        _read_bytes(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            timeout_seconds=10,
            max_bytes=15_000_000,
            user_agent=SEC_USER_AGENT,
        )
    )
    concepts = ((payload or {}).get("facts") or {}).get("us-gaap") or {}

    def records(names: tuple[str, ...], *, annual: bool) -> list[dict[str, Any]]:
        for name in names:
            concept = concepts.get(name) or {}
            units = concept.get("units") or {}
            rows = units.get("USD") or units.get("USD/shares") or []
            filtered = [
                row
                for row in rows
                if row.get("form") in ({"10-K", "20-F", "40-F"} if annual else {"10-K", "10-Q", "20-F", "40-F", "6-K"})
                and (not annual or row.get("fp") == "FY")
                and _number(row.get("val")) is not None
                and row.get("end")
            ]
            if filtered:
                latest_by_end: dict[str, dict[str, Any]] = {}
                for row in filtered:
                    end = str(row["end"])
                    if end not in latest_by_end or str(row.get("filed", "")) > str(latest_by_end[end].get("filed", "")):
                        latest_by_end[end] = row
                return sorted(latest_by_end.values(), key=lambda row: str(row["end"]))
        return []

    revenue = records(
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ),
        annual=True,
    )
    net_income = records(("NetIncomeLoss", "ProfitLoss"), annual=True)
    gross_profit = records(("GrossProfit",), annual=True)
    operating_cash = records(("NetCashProvidedByUsedInOperatingActivities",), annual=True)
    capital_expenditure = records(
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForAdditionsToPropertyPlantAndEquipment",
        ),
        annual=True,
    )
    assets = records(("Assets",), annual=False)
    liabilities = records(("Liabilities",), annual=False)
    equity = records(
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        annual=False,
    )

    def latest_value(rows: list[dict[str, Any]]) -> float | None:
        return _number(rows[-1].get("val")) if rows else None

    def growth(rows: list[dict[str, Any]]) -> float | None:
        if len(rows) < 2:
            return None
        current = _number(rows[-1].get("val"))
        previous = _number(rows[-2].get("val"))
        if current is None or previous in {None, 0.0}:
            return None
        return current / previous - 1.0

    revenue_value = latest_value(revenue)
    income_value = latest_value(net_income)
    assets_value = latest_value(assets)
    liabilities_value = latest_value(liabilities)
    equity_value = latest_value(equity)
    cash_value = latest_value(operating_cash)
    gross_profit_value = latest_value(gross_profit)
    capital_expenditure_value = latest_value(capital_expenditure)
    report_dates = [
        str(rows[-1].get("end"))
        for rows in (revenue, net_income, assets, liabilities, equity)
        if rows
    ]
    fields: dict[str, Any] = {
        "longName": str((payload or {}).get("entityName") or company_name),
        "reportDate": max(report_dates) if report_dates else "",
        "revenueGrowth": growth(revenue),
        "earningsGrowth": growth(net_income),
        "profitMargins": income_value / revenue_value if income_value is not None and revenue_value else None,
        "grossMargins": gross_profit_value / revenue_value if gross_profit_value is not None and revenue_value else None,
        "returnOnEquity": income_value / equity_value if income_value is not None and equity_value else None,
        "returnOnAssets": income_value / assets_value if income_value is not None and assets_value else None,
        "debtRatio": liabilities_value / assets_value if liabilities_value is not None and assets_value else None,
        "operatingCashFlowMargin": cash_value / revenue_value if cash_value is not None and revenue_value else None,
        "freeCashFlowMargin": (
            (cash_value - capital_expenditure_value) / revenue_value
            if cash_value is not None and capital_expenditure_value is not None and revenue_value
            else None
        ),
        "cashConversion": cash_value / income_value if cash_value is not None and income_value not in {None, 0.0} else None,
        "annualRevenue": revenue_value,
        "annualNetIncome": income_value,
        "totalAssets": assets_value,
        "secCik": cik,
        "_source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
    }
    return {key: value for key, value in fields.items() if value not in {None, ""}}


def _nasdaq_official_fundamentals(symbol: str) -> dict[str, Any]:
    payload = json.loads(
        _read_bytes(
            f"https://api.nasdaq.com/api/company/{symbol}/financials?frequency=1",
            timeout_seconds=10,
            max_bytes=5_000_000,
        )
    )
    data = (payload or {}).get("data") or {}
    income = data.get("incomeStatementTable") or {}
    balance = data.get("balanceSheetTable") or {}

    def amount(value: Any) -> float | None:
        text = str(value or "").strip().replace(",", "")
        if not text or text in {"--", "N/A"}:
            return None
        negative = text.startswith("-$") or (text.startswith("(") and text.endswith(")"))
        text = text.replace("$", "").replace("(", "").replace(")", "")
        number = _number(text)
        return -abs(number) if number is not None and negative else number

    def values(table: dict[str, Any], label: str) -> tuple[float | None, float | None]:
        for row in table.get("rows") or []:
            if str(row.get("value1") or "").strip().casefold() == label.casefold():
                return amount(row.get("value2")), amount(row.get("value3"))
        return None, None

    revenue, previous_revenue = values(income, "Total Revenue")
    gross_profit, _ = values(income, "Gross Profit")
    net_income, previous_income = values(income, "Net Income")
    assets, _ = values(balance, "Total Assets")
    liabilities, _ = values(balance, "Total Liabilities")
    equity, _ = values(balance, "Total Equity")
    current_assets, _ = values(balance, "Total Current Assets")
    current_liabilities, _ = values(balance, "Total Current Liabilities")
    short_debt, _ = values(balance, "Short-Term Debt / Current Portion of Long-Term Debt")
    long_debt, _ = values(balance, "Long-Term Debt")
    total_debt = sum(value or 0.0 for value in (short_debt, long_debt))

    def growth(current: float | None, previous: float | None) -> float | None:
        if current is None or previous in {None, 0.0}:
            return None
        return current / previous - 1.0

    headers = income.get("headers") or balance.get("headers") or {}
    raw_report_date = str(headers.get("value2") or "")
    parsed_report_date = pd.to_datetime(raw_report_date, errors="coerce")
    fields: dict[str, Any] = {
        "reportDate": (
            parsed_report_date.date().isoformat()
            if not pd.isna(parsed_report_date)
            else raw_report_date
        ),
        "revenueGrowth": growth(revenue, previous_revenue),
        "earningsGrowth": growth(net_income, previous_income),
        "profitMargins": net_income / revenue if net_income is not None and revenue else None,
        "grossMargins": gross_profit / revenue if gross_profit is not None and revenue else None,
        "returnOnEquity": net_income / equity if net_income is not None and equity else None,
        "returnOnAssets": net_income / assets if net_income is not None and assets else None,
        "debtRatio": liabilities / assets if liabilities is not None and assets else None,
        "debtToEquityRatio": total_debt / equity if total_debt and equity else None,
        "currentRatio": current_assets / current_liabilities if current_assets is not None and current_liabilities else None,
        "annualRevenue": revenue,
        "annualNetIncome": net_income,
        "totalAssets": assets,
        "_source_url": f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}/financials",
    }
    return {key: value for key, value in fields.items() if value not in {None, ""}}


def _provider_error(error: BaseException) -> str:
    return summarize_error(error, 180)


_FUNDAMENTAL_COMPARISON_GROUPS: dict[str, tuple[tuple[str, float], ...]] = {
    "pe_ttm": (("trailingPE", 1.0), ("peRatio", 1.0)),
    "price_to_book": (("priceToBook", 1.0), ("pbRatio", 1.0)),
    "dividend_yield": (("dividendYield", 1.0), ("yieldPercent", 0.01)),
    "revenue_growth": (("revenueGrowth", 1.0),),
    "earnings_growth": (("earningsGrowth", 1.0), ("netIncomeGrowth", 1.0)),
    "profit_margin": (("profitMargins", 1.0), ("netProfitMargin", 1.0)),
    "gross_margin": (("grossMargins", 1.0),),
    "return_on_equity": (("returnOnEquity", 1.0), ("roe", 1.0)),
    "return_on_assets": (("returnOnAssets", 1.0),),
    "debt_to_equity": (("debtToEquity", 0.01), ("debtToEquityRatio", 1.0)),
    "debt_ratio": (("debtRatio", 1.0),),
    "current_ratio": (("currentRatio", 1.0),),
    "quick_ratio": (("quickRatio", 1.0),),
    "operating_cash_margin": (("operatingCashFlowMargin", 1.0),),
    "free_cash_flow_margin": (("freeCashFlowMargin", 1.0),),
    "cash_conversion": (("cashConversion", 1.0),),
}
_MARKET_VALUE_FIELDS = {
    "marketCap",
    "trailingPE",
    "forwardPE",
    "peRatio",
    "priceToBook",
    "pbRatio",
    "dividendYield",
    "yieldPercent",
}


def _provider_reliability(source_kind: str, credibility: str) -> float:
    base = {
        "regulatory-filing": 1.0,
        "exchange-data": 0.96,
        "financial-statement": 0.88,
        "fundamental-data": 0.76,
        "market-data": 0.72,
    }.get(source_kind, 0.68)
    if any(word in credibility for word in ("监管", "官方", "法定")):
        base = max(base, 0.95)
    return base


def _field_priority(key: str, source_kind: str, credibility: str) -> float:
    reliability = _provider_reliability(source_kind, credibility)
    if key in _MARKET_VALUE_FIELDS:
        if source_kind == "market-data":
            return max(reliability, 0.92)
        if source_kind == "fundamental-data":
            return max(reliability, 0.86)
    return reliability


def _agreement(values: list[Any], *, floor: float = 0.05) -> float:
    numeric = [_number(value) for value in values]
    usable = [value for value in numeric if value is not None]
    if len(usable) >= 2:
        median = float(np.median(usable))
        deviations = [
            abs(value - median) / max(abs(value), abs(median), floor)
            for value in usable
        ]
        return max(0.0, min(1.0, 1.0 - float(np.mean(deviations)) / 0.35))
    if len(usable) == 1:
        return 0.5
    text_values = [str(value).strip().casefold() for value in values if str(value).strip()]
    if len(text_values) >= 2:
        most_common = max(text_values.count(value) for value in set(text_values))
        return most_common / len(text_values)
    return 0.5 if text_values else 0.0


def _build_fundamental_quality(
    evidence_by_field: dict[str, list[dict[str, Any]]],
    merged: dict[str, Any],
    statuses: list[FundamentalProviderStatus],
) -> dict[str, Any]:
    metric_quality: dict[str, dict[str, Any]] = {}
    agreements: list[float] = []
    verified_metrics = 0
    for metric, aliases in _FUNDAMENTAL_COMPARISON_GROUPS.items():
        entries: list[dict[str, Any]] = []
        values: list[Any] = []
        for alias, scale in aliases:
            for entry in evidence_by_field.get(alias, []):
                value = _number(entry.get("value"))
                if value is None:
                    continue
                entries.append(entry)
                values.append(value * scale)
        unique_providers = {str(entry.get("provider") or "") for entry in entries}
        source_count = len(unique_providers)
        agreement = _agreement(values)
        selected_reliability = max(
            (float(entry.get("reliability") or 0.0) for entry in entries),
            default=0.0,
        )
        quality_score = round(
            100
            * (
                0.50 * selected_reliability
                + 0.35 * agreement
                + 0.15 * min(source_count / 2.0, 1.0)
            )
        )
        if entries:
            agreements.append(agreement)
        if source_count >= 2:
            verified_metrics += 1
        metric_quality[metric] = {
            "source_count": source_count,
            "agreement": round(agreement, 6),
            "quality_score": quality_score,
            "official_source_count": sum(
                float(entry.get("reliability") or 0.0) >= 0.95
                for entry in entries
            ),
        }

    ok_statuses = [status for status in statuses if status.status == "ok"]
    provider_ratio = len(ok_statuses) / len(statuses) if statuses else 0.0
    source_trust = (
        float(
            np.mean(
                [
                    _provider_reliability(status.source_kind, status.credibility)
                    for status in ok_statuses
                ]
            )
        )
        if ok_statuses
        else 0.0
    )
    report_date = str(merged.get("reportDate") or "")[:10]
    age_days: int | None = None
    freshness = 0.72
    if report_date:
        try:
            age_days = max(
                0,
                (datetime.now(timezone.utc).date() - pd.Timestamp(report_date).date()).days,
            )
            freshness = max(0.35, min(1.0, 1.0 - max(0, age_days - 120) / 720))
        except (TypeError, ValueError):
            freshness = 0.65
    agreement_mean = float(np.mean(agreements)) if agreements else 0.0
    comparable_count = sum(
        bool(metric_quality[metric]["source_count"])
        for metric in _FUNDAMENTAL_COMPARISON_GROUPS
    )
    verification_coverage = (
        verified_metrics / comparable_count if comparable_count else 0.0
    )
    quality_score = round(
        100
        * (
            0.30 * provider_ratio
            + 0.25 * source_trust
            + 0.20 * agreement_mean
            + 0.15 * freshness
            + 0.10 * verification_coverage
        )
    )
    label = (
        "高可信"
        if quality_score >= 80
        else "较可信"
        if quality_score >= 65
        else "需核验"
        if quality_score >= 45
        else "证据不足"
    )
    return {
        "score": quality_score,
        "label": label,
        "provider_coverage": round(provider_ratio, 6),
        "source_trust": round(source_trust, 6),
        "cross_source_agreement": round(agreement_mean, 6),
        "verified_metric_count": verified_metrics,
        "comparable_metric_count": comparable_count,
        "verification_coverage": round(verification_coverage, 6),
        "report_age_days": age_days,
        "freshness": round(freshness, 6),
        "metric_quality": metric_quality,
        "methodology": "字段级保留来源原值，官方/交易所/财报优先，按跨源偏差、报告期和来源等级计算质量。",
    }


def fetch_fundamental_snapshot(
    market: str,
    symbol: str,
    *,
    timeout_seconds: int = 12,
) -> FundamentalSnapshot:
    normalized_market = market.strip().lower()
    normalized_symbol = symbol.strip().upper()
    if normalized_market == "demo":
        fields = {
            "longName": f"Demo Company {normalized_symbol}",
            "marketCap": 50_000_000_000,
            "trailingPE": 22.0,
            "priceToBook": 3.1,
            "dividendYield": 0.018,
            "profitMargins": 0.16,
            "revenueGrowth": 0.09,
            "earningsGrowth": 0.11,
            "debtToEquity": 48.0,
            "returnOnEquity": 0.17,
            "currentRatio": 1.8,
            "reportDate": datetime.now().date().isoformat(),
        }
        label = "内置模拟基本面"
        return FundamentalSnapshot(
            market=normalized_market,
            symbol=normalized_symbol,
            fields=fields,
            field_sources={key: label for key in fields},
            providers=(FundamentalProviderStatus("demo", label, "ok", len(fields)),),
            warnings=("模拟基本面仅用于离线学习。",),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    provider_specs: tuple[tuple[str, str, str, str, str, Any], ...]
    if normalized_market == "a-share":
        normalized_symbol = normalize_a_share_symbol(normalized_symbol)
        provider_specs = (
            (
                "msn-finance",
                "Microsoft Finance",
                "https://www.msn.com/en-us/money",
                "market-data",
                "公开行情与估值",
                lambda: _msn_fundamentals(normalized_market, normalized_symbol),
            ),
            (
                "sina-financials",
                "新浪财经财务指标",
                "https://finance.sina.com.cn/stock/",
                "financial-statement",
                "上市公司财务指标",
                lambda: _a_share_financials(normalized_symbol),
            ),
            (
                "eastmoney-financials",
                "东方财富财务分析",
                "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html#/cwfx",
                "financial-statement",
                "公开财务报表指标",
                lambda: _eastmoney_financial_analysis(
                    normalized_market, normalized_symbol
                ),
            ),
        )
    elif normalized_market in {"nasdaq", "hk", "global"}:
        provider_list: list[tuple[str, str, str, str, str, Any]] = [
            (
                "msn-finance",
                "Microsoft Finance",
                "https://www.msn.com/en-us/money",
                "market-data",
                "公开行情与估值",
                lambda: _msn_fundamentals(normalized_market, normalized_symbol),
            ),
            (
                "yahoo-fundamentals",
                "Yahoo Finance 基本面",
                f"https://finance.yahoo.com/quote/{normalized_symbol}",
                "fundamental-data",
                "公开公司数据",
                lambda: _yfinance_fundamentals(normalized_symbol),
            ),
        ]
        if normalized_market in {"nasdaq", "hk"}:
            provider_list.append(
                (
                    "eastmoney-financials",
                    "东方财富财务分析",
                    "https://emweb.eastmoney.com/",
                    "financial-statement",
                    "公开财务报表指标",
                    lambda: _eastmoney_financial_analysis(
                        normalized_market, normalized_symbol
                    ),
                )
            )
        if normalized_market == "nasdaq":
            provider_list.append(
                (
                    "nasdaq-financials",
                    "Nasdaq 官方财务",
                    f"https://www.nasdaq.com/market-activity/stocks/{normalized_symbol.lower()}/financials",
                    "exchange-data",
                    "交易所官方",
                    lambda: _nasdaq_official_fundamentals(normalized_symbol),
                )
            )
            provider_list.append(
                (
                    "sec-companyfacts",
                    "SEC XBRL Company Facts",
                    "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                    "regulatory-filing",
                    "监管机构官方",
                    lambda: _sec_company_fundamentals(normalized_symbol),
                )
            )
        provider_specs = tuple(provider_list)
    else:
        return FundamentalSnapshot(
            market=normalized_market,
            symbol=normalized_symbol,
            fields={},
            field_sources={},
            providers=(),
            warnings=("当前市场不自动获取基本面。",),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    results: queue.Queue[tuple[str, dict[str, Any] | None, BaseException | None]] = queue.Queue()

    def run_provider(provider: str, loader) -> None:
        try:
            results.put((provider, loader(), None))
        except Exception as error:
            results.put((provider, None, error))

    for provider, _label, _url, _kind, _credibility, loader in provider_specs:
        threading.Thread(
            target=run_provider,
            args=(provider, loader),
            daemon=True,
            name=f"fundamentals-{provider}",
        ).start()

    deadline = time.monotonic() + max(2.0, timeout_seconds)
    completed: dict[str, tuple[dict[str, Any] | None, BaseException | None]] = {}
    while len(completed) < len(provider_specs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            provider, fields, error = results.get(timeout=remaining)
        except queue.Empty:
            break
        completed[provider] = (fields, error)

    merged: dict[str, Any] = {}
    field_sources: dict[str, str] = {}
    evidence_by_field: dict[str, list[dict[str, Any]]] = {}
    statuses: list[FundamentalProviderStatus] = []
    warnings: list[str] = []
    for provider, label, source_url, source_kind, credibility, _loader in provider_specs:
        if provider not in completed:
            message = f"{label} 获取超时"
            statuses.append(
                FundamentalProviderStatus(
                    provider, label, "timeout", 0, message, source_url, source_kind, credibility
                )
            )
            warnings.append(message)
            continue
        fields, error = completed[provider]
        if error is not None:
            message = _provider_error(error)
            statuses.append(
                FundamentalProviderStatus(
                    provider, label, "error", 0, message, source_url, source_kind, credibility
                )
            )
            warnings.append(f"{label} 暂不可用：{message}")
            continue
        usable = dict(fields or {})
        dynamic_url = str(usable.pop("_source_url", "") or source_url)
        usable = {key: value for key, value in usable.items() if value is not None}
        for key, value in usable.items():
            reliability = _field_priority(key, source_kind, credibility)
            evidence_by_field.setdefault(key, []).append(
                {
                    "provider": provider,
                    "label": label,
                    "value": value,
                    "source_url": dynamic_url,
                    "source_kind": source_kind,
                    "credibility": credibility,
                    "reliability": round(reliability, 4),
                }
            )
        statuses.append(
            FundamentalProviderStatus(
                provider,
                label,
                "ok" if usable else "empty",
                len(usable),
                "" if usable else "未返回可用字段",
                dynamic_url,
                source_kind,
                credibility,
            )
        )
    field_evidence: dict[str, tuple[dict[str, Any], ...]] = {}
    for key, entries in evidence_by_field.items():
        ordered = sorted(
            entries,
            key=lambda entry: (
                float(entry.get("reliability") or 0.0),
                str(entry.get("label") or ""),
            ),
            reverse=True,
        )
        selected = ordered[0]
        merged[key] = selected["value"]
        field_sources[key] = str(selected["label"])
        field_evidence[key] = tuple(ordered)
    if not merged:
        warnings.append("基本面数据为空，本地评分已自动排除该证据。")
    quality = _build_fundamental_quality(evidence_by_field, merged, statuses)
    if quality["cross_source_agreement"] < 0.55 and quality["verified_metric_count"]:
        warnings.append("部分基本面字段跨源偏差较大，本地评分已降低其权重。")
    return FundamentalSnapshot(
        market=normalized_market,
        symbol=normalized_symbol,
        fields=merged,
        field_sources=field_sources,
        providers=tuple(statuses),
        warnings=tuple(warnings),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        field_evidence=field_evidence,
        quality=quality,
    )


def fetch_research_evidence(
    source: str,
    symbol: str,
    *,
    timeout_seconds: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Fetch optional company and news evidence without blocking the workflow forever."""

    normalized_source = source.strip().lower()
    warnings: list[str] = []
    try:
        if normalized_source in {"a-share", "nasdaq", "hk", "global"}:
            normalized_symbol = (
                normalize_a_share_symbol(symbol)
                if normalized_source == "a-share"
                else symbol.strip().upper()
            )
            snapshot = _run_with_timeout(
                lambda: fetch_fundamental_snapshot(
                    normalized_source,
                    normalized_symbol,
                    timeout_seconds=max(3, timeout_seconds - 1),
                ),
                timeout_seconds,
            )
            if not isinstance(snapshot, FundamentalSnapshot):
                raise RuntimeError("基本面快照返回格式无效。")
            fundamentals = dict(snapshot.fields)
            fundamentals["_evidence_quality"] = snapshot.quality
            fundamentals["_field_sources"] = snapshot.field_sources
            news: list[dict[str, str]] = []
            warnings.extend(snapshot.warnings)
        else:
            return {}, [], [f"{normalized_source} 模式不自动获取基本面和新闻。"]
    except Exception as exc:
        market_label = {
            "a-share": "A股",
            "nasdaq": "美股",
            "hk": "港股",
            "global": "全球市场",
        }.get(normalized_source, normalized_source)
        warnings.append(
            f"{market_label}基本面/新闻暂不可用：" + summarize_error(exc, 220)
        )
        return {}, [], warnings

    if not fundamentals:
        warnings.append("基本面数据为空，基本面分析师将明确标记数据缺口。")
    if not news:
        warnings.append("新闻数据为空，新闻与情绪分析师将仅使用可得证据。")
    return fundamentals, news, warnings


def build_research_context(
    source: str,
    symbol: str,
    start: str,
    end: str,
    *,
    adjust: str = "qfq",
    auto_adjust: bool = True,
    csv_path: str | Path | None = None,
    seed: int = 7,
    fetch_details: bool = True,
) -> ResearchContext:
    """Load bars and optional company/news evidence without hiding failures."""

    warnings: list[str] = []
    normalized_source = source.strip().lower()

    if normalized_source == "demo":
        normalized_symbol = symbol.strip().upper() or "ALPHA"
        if normalized_symbol not in DemoMarketConfig().tickers:
            normalized_symbol = "ALPHA"
        bars = generate_demo_ohlcv(
            normalized_symbol,
            DemoMarketConfig(start=start, end=end, seed=seed),
        )
        fundamentals = {
            "company": f"Demo Company {normalized_symbol}",
            "marketCap": 50_000_000_000,
            "trailingPE": 22.0,
            "profitMargins": 0.16,
            "revenueGrowth": 0.09,
            "debtToEquity": 48.0,
            "note": "模拟数据，仅用于离线学习。",
        }
        news = [
            {"title": "公司发布稳健季度经营数据", "publisher": "Demo News"},
            {"title": "行业需求预期保持温和增长", "publisher": "Demo News"},
            {"title": "市场关注估值与短期波动风险", "publisher": "Demo News"},
        ]
        warnings.append("当前使用模拟行情、模拟基本面和模拟新闻，仅用于离线学习。")
    elif normalized_source == "a-share":
        normalized_symbol = normalize_a_share_symbol(symbol)
        bars = fetch_a_share_ohlcv(normalized_symbol, start, end, adjust)
        fundamentals, news = {}, []
        if fetch_details:
            fundamentals, news, detail_warnings = fetch_research_evidence(
                normalized_source, normalized_symbol
            )
            warnings.extend(detail_warnings)
    elif normalized_source == "nasdaq":
        normalized_symbol = symbol.strip().upper()
        bars = fetch_nasdaq_ohlcv(normalized_symbol, start, end, auto_adjust)
        fundamentals, news = {}, []
        if fetch_details:
            fundamentals, news, detail_warnings = fetch_research_evidence(
                normalized_source, normalized_symbol
            )
            warnings.extend(detail_warnings)
    elif normalized_source == "hk":
        normalized_symbol = normalize_hk_symbol(symbol)
        bars = fetch_hk_ohlcv(normalized_symbol, start, end, adjust)
        fundamentals, news = {}, []
        if fetch_details:
            fundamentals, news, detail_warnings = fetch_research_evidence(
                normalized_source, normalized_symbol
            )
            warnings.extend(detail_warnings)
    elif normalized_source == "global":
        normalized_symbol = symbol.strip().upper()
        bars = fetch_global_ohlcv(normalized_symbol, start, end, auto_adjust)
        fundamentals, news = {}, []
        if fetch_details:
            fundamentals, news, detail_warnings = fetch_research_evidence(
                normalized_source, normalized_symbol
            )
            warnings.extend(detail_warnings)
    elif normalized_source == "csv":
        if not csv_path:
            raise ValueError("CSV 数据源必须选择 OHLCV 文件。")
        normalized_symbol = symbol.strip().upper() or "CSV"
        bars = load_ohlcv_csv(csv_path)
        fundamentals, news = {}, []
        warnings.append("CSV 模式不自动获取基本面和新闻。")
    else:
        raise ValueError(f"未知数据源：{source}")

    if len(bars) < 60:
        warnings.append("数据少于 60 行，SMA60 与中期指标可能为空。")

    provider = str(bars.attrs.get("provider", normalized_source))
    if provider in {"tencent-direct", "akshare-tencent"}:
        warnings.append("东方财富接口不可用，本次 A 股行情改由腾讯行情提供。")
    elif provider == "stooq":
        warnings.append("Yahoo Finance 限流，本次美股行情改由 Stooq 提供。")
    if bars.attrs.get("adjustment_fallback"):
        warnings.append("腾讯未返回所请求的复权键，本次使用其可用日线字段。")
    technical = calculate_technical_snapshot(bars)
    technical["data_provider"] = provider
    technical["data_provider_label"] = provider_display_name(provider)

    return ResearchContext(
        symbol=normalized_symbol,
        market=normalized_source,
        analysis_date=end,
        bars=bars,
        technical=technical,
        fundamentals=fundamentals,
        news=news,
        warnings=warnings,
    )
