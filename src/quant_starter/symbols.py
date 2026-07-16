from __future__ import annotations

from dataclasses import dataclass
import re

from .data import DemoMarketConfig, normalize_a_share_symbol, normalize_hk_symbol


@dataclass(frozen=True)
class StockPreset:
    source: str
    symbol: str
    name: str
    asset_type: str = "stock"
    exchange: str = ""
    country: str = ""
    currency: str = ""
    category: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} · {self.symbol}"


STOCK_PRESETS = (
    StockPreset("demo", "ALPHA", "Alpha 科技"),
    StockPreset("demo", "BALANCE", "Balance 均衡"),
    StockPreset("demo", "CYCLE", "Cycle 周期"),
    StockPreset("demo", "DEFENSE", "Defense 防御"),
    StockPreset("demo", "GROWTH", "Growth 成长"),
    StockPreset("demo", "VALUE", "Value 价值"),
    StockPreset("a-share", "600519", "贵州茅台", exchange="上交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "300750", "宁德时代", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "002594", "比亚迪", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "601318", "中国平安", exchange="上交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "600036", "招商银行", exchange="上交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "000858", "五粮液", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "300059", "东方财富", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "600030", "中信证券", exchange="上交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "601398", "工商银行", exchange="上交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "688981", "中芯国际", exchange="科创板", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "000651", "格力电器", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "000001", "平安银行", exchange="深交所", country="CN", currency="CNY", category="沪深股票"),
    StockPreset("a-share", "510300", "沪深300ETF", "etf", "上交所", "CN", "CNY", "宽基 ETF"),
    StockPreset("a-share", "510500", "中证500ETF", "etf", "上交所", "CN", "CNY", "宽基 ETF"),
    StockPreset("a-share", "159915", "创业板ETF", "etf", "深交所", "CN", "CNY", "宽基 ETF"),
    StockPreset("a-share", "512480", "半导体ETF", "etf", "上交所", "CN", "CNY", "行业 ETF"),
    StockPreset("a-share", "512800", "银行ETF", "etf", "上交所", "CN", "CNY", "行业 ETF"),
    StockPreset("a-share", "513100", "纳指ETF", "etf", "上交所", "CN", "CNY", "跨境 ETF"),
    StockPreset("a-share", "513500", "标普500ETF", "etf", "上交所", "CN", "CNY", "跨境 ETF"),
    StockPreset("a-share", "518880", "黄金ETF", "etf", "上交所", "CN", "CNY", "商品 ETF"),
    StockPreset("a-share", "511010", "国债ETF", "etf", "上交所", "CN", "CNY", "债券 ETF"),
    StockPreset("nasdaq", "AAPL", "Apple", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "MSFT", "Microsoft", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "NVDA", "NVIDIA", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "AMZN", "Amazon", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "GOOGL", "Alphabet", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "META", "Meta", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "TSLA", "Tesla", exchange="NASDAQ", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "JPM", "JPMorgan", exchange="NYSE", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "BRK-B", "Berkshire Hathaway", exchange="NYSE", country="US", currency="USD", category="美国股票"),
    StockPreset("nasdaq", "SPY", "SPDR S&P 500 ETF", "etf", "NYSE ARCA", "US", "USD", "宽基 ETF"),
    StockPreset("nasdaq", "QQQ", "Invesco Nasdaq 100 ETF", "etf", "NASDAQ", "US", "USD", "宽基 ETF"),
    StockPreset("nasdaq", "IWM", "iShares Russell 2000 ETF", "etf", "NYSE ARCA", "US", "USD", "宽基 ETF"),
    StockPreset("nasdaq", "VTI", "Vanguard Total Stock Market ETF", "etf", "NYSE ARCA", "US", "USD", "宽基 ETF"),
    StockPreset("nasdaq", "TQQQ", "ProShares UltraPro QQQ", "etf", "NASDAQ", "US", "USD", "杠杆 ETF"),
    StockPreset("nasdaq", "SQQQ", "ProShares UltraPro Short QQQ", "etf", "NASDAQ", "US", "USD", "反向 ETF"),
    StockPreset("nasdaq", "UPRO", "ProShares UltraPro S&P 500", "etf", "NYSE ARCA", "US", "USD", "杠杆 ETF"),
    StockPreset("nasdaq", "SPXU", "ProShares UltraPro Short S&P 500", "etf", "NYSE ARCA", "US", "USD", "反向 ETF"),
    StockPreset("nasdaq", "XLK", "Technology Select Sector SPDR", "etf", "NYSE ARCA", "US", "USD", "行业 ETF"),
    StockPreset("nasdaq", "TLT", "iShares 20+ Year Treasury Bond ETF", "etf", "NASDAQ", "US", "USD", "债券 ETF"),
    StockPreset("nasdaq", "BIL", "SPDR 1-3 Month T-Bill ETF", "etf", "NYSE ARCA", "US", "USD", "债券 ETF"),
    StockPreset("nasdaq", "GLD", "SPDR Gold Shares", "etf", "NYSE ARCA", "US", "USD", "商品 ETF"),
    StockPreset("nasdaq", "EEM", "iShares Emerging Markets ETF", "etf", "NYSE ARCA", "US", "USD", "国际 ETF"),
    StockPreset("nasdaq", "EWJ", "iShares MSCI Japan ETF", "etf", "NYSE ARCA", "US", "USD", "国际 ETF"),
    StockPreset("nasdaq", "IBIT", "iShares Bitcoin Trust ETF", "etf", "NASDAQ", "US", "USD", "数字资产 ETF"),
    StockPreset("hk", "0700.HK", "腾讯控股", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "9988.HK", "阿里巴巴-W", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "3690.HK", "美团-W", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "1810.HK", "小米集团-W", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "1211.HK", "比亚迪股份", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "1299.HK", "友邦保险", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "0941.HK", "中国移动", exchange="港交所", country="HK", currency="HKD", category="香港股票"),
    StockPreset("hk", "2800.HK", "盈富基金", "etf", "港交所", "HK", "HKD", "宽基 ETF"),
    StockPreset("hk", "3033.HK", "恒生科技ETF", "etf", "港交所", "HK", "HKD", "科技 ETF"),
    StockPreset("hk", "2822.HK", "南方A50ETF", "etf", "港交所", "HK", "HKD", "跨境 ETF"),
    StockPreset("hk", "2840.HK", "SPDR金ETF", "etf", "港交所", "HK", "HKD", "商品 ETF"),
    StockPreset("hk", "3110.HK", "恒生高股息率ETF", "etf", "港交所", "HK", "HKD", "策略 ETF"),
    StockPreset("hk", "7226.HK", "南方两倍做多恒科", "etf", "港交所", "HK", "HKD", "杠杆 ETF"),
    StockPreset("hk", "7500.HK", "南方两倍做空恒指", "etf", "港交所", "HK", "HKD", "反向 ETF"),
    StockPreset("global", "7203.T", "Toyota Motor", exchange="东京证券交易所", country="JP", currency="JPY", category="日本"),
    StockPreset("global", "6758.T", "Sony Group", exchange="东京证券交易所", country="JP", currency="JPY", category="日本"),
    StockPreset("global", "9984.T", "SoftBank Group", exchange="东京证券交易所", country="JP", currency="JPY", category="日本"),
    StockPreset("global", "SHEL.L", "Shell", exchange="伦敦证券交易所", country="GB", currency="GBP", category="英国"),
    StockPreset("global", "AZN.L", "AstraZeneca", exchange="伦敦证券交易所", country="GB", currency="GBP", category="英国"),
    StockPreset("global", "SAP.DE", "SAP", exchange="Xetra", country="DE", currency="EUR", category="德国"),
    StockPreset("global", "ASML.AS", "ASML", exchange="阿姆斯特丹交易所", country="NL", currency="EUR", category="荷兰"),
    StockPreset("global", "MC.PA", "LVMH", exchange="巴黎泛欧交易所", country="FR", currency="EUR", category="法国"),
    StockPreset("global", "SHOP.TO", "Shopify", exchange="多伦多证券交易所", country="CA", currency="CAD", category="加拿大"),
    StockPreset("global", "BHP.AX", "BHP Group", exchange="澳大利亚证券交易所", country="AU", currency="AUD", category="澳大利亚"),
    StockPreset("global", "RELIANCE.NS", "Reliance Industries", exchange="印度国家证券交易所", country="IN", currency="INR", category="印度"),
    StockPreset("global", "D05.SI", "DBS Group", exchange="新加坡交易所", country="SG", currency="SGD", category="新加坡"),
    StockPreset("global", "CSPX.L", "iShares Core S&P 500 UCITS ETF", "etf", "伦敦证券交易所", "GB", "USD", "UCITS ETF"),
    StockPreset("global", "VWRL.L", "Vanguard FTSE All-World UCITS ETF", "etf", "伦敦证券交易所", "GB", "USD", "UCITS ETF"),
    StockPreset("global", "EQQQ.L", "Invesco Nasdaq-100 UCITS ETF", "etf", "伦敦证券交易所", "GB", "USD", "UCITS ETF"),
    StockPreset("global", "1306.T", "NEXT FUNDS TOPIX ETF", "etf", "东京证券交易所", "JP", "JPY", "日本 ETF"),
    StockPreset("global", "N225.T", "Nikkei 225", "index", "东京证券交易所", "JP", "JPY", "全球指数"),
)


MARKET_DEFINITIONS = {
    "a-share": {"label": "A 股", "default_symbol": "300750", "currency": "CNY"},
    "nasdaq": {"label": "美 股", "default_symbol": "NVDA", "currency": "USD"},
    "hk": {"label": "港 股", "default_symbol": "0700.HK", "currency": "HKD"},
    "global": {"label": "全 球", "default_symbol": "7203.T", "currency": ""},
    "demo": {"label": "模拟", "default_symbol": "ALPHA", "currency": "USD"},
}


def presets_for_source(
    source: str,
    asset_type: str = "",
) -> tuple[StockPreset, ...]:
    normalized = source.strip().lower()
    normalized = {"us": "nasdaq", "usa": "nasdaq", "world": "global"}.get(
        normalized, normalized
    )
    normalized_asset = asset_type.strip().lower()
    return tuple(
        preset
        for preset in STOCK_PRESETS
        if preset.source == normalized
        and (not normalized_asset or normalized_asset == "all" or preset.asset_type == normalized_asset)
    )


def stock_choice_labels(source: str) -> tuple[str, ...]:
    return tuple(preset.label for preset in presets_for_source(source))


def default_stock_choice(source: str) -> str:
    presets = presets_for_source(source)
    if presets:
        return presets[0].label
    return "CSV"


def display_for_symbol(source: str, symbol: str) -> str:
    normalized_symbol = symbol.strip().upper()
    for preset in presets_for_source(source):
        if preset.symbol == normalized_symbol:
            return preset.label
    return normalized_symbol


def preset_for_symbol(source: str, symbol: str) -> StockPreset | None:
    normalized_symbol = symbol.strip().upper()
    return next(
        (
            preset
            for preset in presets_for_source(source)
            if preset.symbol == normalized_symbol
        ),
        None,
    )


def resolve_stock_choice(value: str, source: str) -> str:
    """Resolve a preset label or a manually entered symbol."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("请选择个股或输入股票代码。")
    normalized_source = source.strip().lower()

    for preset in presets_for_source(normalized_source):
        if cleaned.casefold() in {preset.label.casefold(), preset.name.casefold()}:
            return preset.symbol

    if normalized_source == "a-share":
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", cleaned)
        if match is None:
            raise ValueError("A 股请输入 6 位股票代码，例如 600519。")
        return normalize_a_share_symbol(match.group(1))

    if normalized_source == "demo":
        symbol = cleaned.upper().split("·")[-1].strip()
        if symbol not in DemoMarketConfig().tickers:
            raise ValueError(
                "模拟标的可选：" + "、".join(DemoMarketConfig().tickers)
            )
        return symbol

    if normalized_source in {"nasdaq", "us", "usa"}:
        label_match = re.search(r"·\s*([A-Za-z0-9.^=-]+)\s*$", cleaned)
        symbol = label_match.group(1) if label_match else cleaned
        symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", symbol):
            raise ValueError("美股请输入有效代码，例如 AAPL、NVDA 或 QQQ。")
        return symbol

    if normalized_source in {"hk", "hong-kong"}:
        label_match = re.search(r"·\s*([0-9]{1,5}(?:\.HK)?)\s*$", cleaned, re.IGNORECASE)
        symbol = label_match.group(1) if label_match else cleaned
        return normalize_hk_symbol(symbol)

    if normalized_source in {"global", "world", "international"}:
        label_match = re.search(r"·\s*([A-Za-z0-9.^=-]+)\s*$", cleaned)
        symbol = (label_match.group(1) if label_match else cleaned).strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,24}", symbol):
            raise ValueError("全球证券请输入有效交易所代码，例如 7203.T、SHEL.L 或 CSPX.L。")
        return symbol

    if normalized_source == "csv":
        return cleaned.upper()
    raise ValueError(f"未知数据源：{source}")


def market_definition(source: str) -> dict[str, str]:
    normalized = source.strip().lower()
    normalized = {"us": "nasdaq", "usa": "nasdaq", "world": "global"}.get(
        normalized, normalized
    )
    if normalized not in MARKET_DEFINITIONS:
        raise ValueError(f"未知市场：{source}")
    return {"id": normalized, **MARKET_DEFINITIONS[normalized]}


def search_local_presets(
    query: str,
    source: str,
    asset_type: str = "",
    limit: int = 12,
) -> tuple[StockPreset, ...]:
    needle = query.strip().casefold()
    matches = [
        item
        for item in presets_for_source(source, asset_type)
        if not needle
        or needle in item.symbol.casefold()
        or needle in item.name.casefold()
        or needle in item.category.casefold()
    ]
    return tuple(matches[: max(1, min(int(limit), 30))])
