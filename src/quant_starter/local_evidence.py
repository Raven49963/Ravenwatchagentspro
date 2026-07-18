from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Callable


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, lower: float = -100.0, upper: float = 100.0) -> float:
    return min(upper, max(lower, value))


def _rating_score(directional_score: float) -> float:
    return _clamp(50.0 + _clamp(directional_score) / 2.0, 0.0, 100.0)


def _direction_phrase(score: float) -> str:
    if score >= 25:
        return "明显偏多"
    if score >= 8:
        return "略偏多"
    if score > -8:
        return "中性"
    if score > -25:
        return "略偏空"
    return "明显偏空"


def _piecewise(value: float, points: tuple[tuple[float, float], ...]) -> float:
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= value <= right_x:
            span = right_x - left_x
            ratio = 0.0 if span == 0 else (value - left_x) / span
            return left_y + ratio * (right_y - left_y)
    return 0.0


def _metric_tone(score: float) -> str:
    if score >= 20:
        return "positive"
    if score <= -20:
        return "negative"
    return "neutral"


def _extract(
    fields: dict[str, Any],
    field_sources: dict[str, str],
    aliases: tuple[tuple[str, float], ...],
) -> tuple[float | None, str, str]:
    for key, scale in aliases:
        value = _finite(fields.get(key))
        if value is not None:
            return value * scale, field_sources.get(key, ""), key
    return None, "", ""


def _format_metric(value: float | None, unit: str) -> str:
    if value is None:
        return "--"
    if unit == "percent":
        return f"{value * 100:.1f}%"
    if unit == "multiple":
        return f"{value:.2f}x"
    if unit == "date":
        return str(value)
    return f"{value:,.2f}"


_FUNDAMENTAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "pe_ttm",
        "label": "市盈率 TTM",
        "category": "valuation",
        "unit": "multiple",
        "aliases": (("trailingPE", 1.0), ("peRatio", 1.0), ("市盈率(动态)", 1.0)),
        "score": lambda value: _piecewise(
            value, ((0, -80), (8, 42), (15, 60), (25, 28), (40, -20), (80, -75))
        ),
    },
    {
        "key": "price_to_book",
        "label": "市净率",
        "category": "valuation",
        "unit": "multiple",
        "aliases": (("priceToBook", 1.0), ("pbRatio", 1.0), ("市净率", 1.0)),
        "score": lambda value: _piecewise(
            value, ((0, -70), (0.8, 55), (2, 42), (4, 8), (8, -45), (15, -75))
        ),
    },
    {
        "key": "dividend_yield",
        "label": "股息率",
        "category": "valuation",
        "unit": "percent",
        "aliases": (("dividendYield", 1.0), ("yieldPercent", 0.01), ("股息率", 0.01)),
        "score": lambda value: _piecewise(
            value, ((0, -5), (0.015, 18), (0.035, 48), (0.07, 58), (0.12, 0), (0.25, -55))
        ),
    },
    {
        "key": "revenue_growth",
        "label": "营收增长",
        "category": "growth",
        "unit": "percent",
        "aliases": (("revenueGrowth", 1.0), ("主营业务收入增长率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((-0.5, -90), (-0.1, -42), (0, 0), (0.1, 38), (0.25, 70), (0.6, 58), (1.5, 25))
        ),
    },
    {
        "key": "earnings_growth",
        "label": "净利润增长",
        "category": "growth",
        "unit": "percent",
        "aliases": (("earningsGrowth", 1.0), ("netIncomeGrowth", 1.0), ("净利润增长率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((-0.8, -95), (-0.15, -48), (0, 0), (0.12, 38), (0.3, 72), (0.8, 58), (2, 22))
        ),
    },
    {
        "key": "profit_margin",
        "label": "净利率",
        "category": "profitability",
        "unit": "percent",
        "aliases": (("profitMargins", 1.0), ("netProfitMargin", 1.0), ("销售净利率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((-0.3, -95), (0, -35), (0.05, 5), (0.15, 48), (0.3, 72), (0.6, 78))
        ),
    },
    {
        "key": "gross_margin",
        "label": "毛利率",
        "category": "profitability",
        "unit": "percent",
        "aliases": (("grossMargins", 1.0),),
        "score": lambda value: _piecewise(
            value, ((-0.1, -90), (0, -35), (0.15, 8), (0.3, 38), (0.5, 62), (0.8, 72))
        ),
    },
    {
        "key": "return_on_equity",
        "label": "净资产收益率",
        "category": "profitability",
        "unit": "percent",
        "aliases": (("returnOnEquity", 1.0), ("roe", 1.0), ("净资产收益率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((-0.4, -95), (0, -28), (0.08, 12), (0.15, 48), (0.3, 72), (0.6, 52), (1.2, 18))
        ),
    },
    {
        "key": "return_on_assets",
        "label": "总资产回报率",
        "category": "profitability",
        "unit": "percent",
        "aliases": (("returnOnAssets", 1.0),),
        "score": lambda value: _piecewise(
            value, ((-0.2, -92), (0, -25), (0.04, 8), (0.08, 38), (0.15, 68), (0.35, 76))
        ),
    },
    {
        "key": "debt_to_equity",
        "label": "负债权益比",
        "category": "leverage",
        "unit": "percent",
        "aliases": (("debtToEquity", 0.01), ("debtToEquityRatio", 1.0), ("负债与所有者权益比率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((0, 62), (0.4, 55), (0.8, 28), (1.5, -12), (3, -68), (6, -92))
        ),
    },
    {
        "key": "debt_ratio",
        "label": "资产负债率",
        "category": "leverage",
        "unit": "percent",
        "aliases": (("debtRatio", 1.0), ("资产负债率(%)", 0.01)),
        "score": lambda value: _piecewise(
            value, ((0, 58), (0.3, 52), (0.5, 18), (0.7, -30), (0.9, -82), (1.2, -95))
        ),
    },
    {
        "key": "current_ratio",
        "label": "流动比率",
        "category": "leverage",
        "unit": "multiple",
        "aliases": (("currentRatio", 1.0), ("流动比率", 1.0)),
        "score": lambda value: _piecewise(
            value, ((0.3, -85), (0.8, -42), (1, -8), (1.5, 42), (3, 55), (6, 20), (12, -5))
        ),
    },
    {
        "key": "quick_ratio",
        "label": "速动比率",
        "category": "leverage",
        "unit": "multiple",
        "aliases": (("quickRatio", 1.0),),
        "score": lambda value: _piecewise(
            value, ((0.2, -85), (0.6, -48), (1, 5), (1.5, 42), (2.5, 55), (6, 15), (12, -8))
        ),
    },
    {
        "key": "operating_cash_margin",
        "label": "经营现金流率",
        "category": "cashflow",
        "unit": "percent",
        "aliases": (("operatingCashFlowMargin", 1.0),),
        "score": lambda value: _piecewise(
            value, ((-0.3, -95), (0, -28), (0.05, 8), (0.15, 48), (0.3, 72), (0.7, 78))
        ),
    },
    {
        "key": "free_cash_flow_margin",
        "label": "自由现金流率",
        "category": "cashflow",
        "unit": "percent",
        "aliases": (("freeCashFlowMargin", 1.0),),
        "score": lambda value: _piecewise(
            value, ((-0.3, -95), (0, -32), (0.04, 8), (0.12, 45), (0.25, 70), (0.6, 76))
        ),
    },
    {
        "key": "cash_conversion",
        "label": "现金利润比",
        "category": "cashflow",
        "unit": "multiple",
        "aliases": (("cashConversion", 1.0),),
        "score": lambda value: _piecewise(
            value, ((-2, -95), (0, -45), (0.5, 0), (0.9, 42), (1.3, 64), (2.5, 48), (5, 5))
        ),
    },
)


_CATEGORY_LABELS = {
    "valuation": "估值",
    "growth": "增长",
    "profitability": "盈利",
    "leverage": "偿债",
    "cashflow": "现金流",
}
_CATEGORY_WEIGHTS = {
    "valuation": 0.20,
    "growth": 0.20,
    "profitability": 0.25,
    "leverage": 0.15,
    "cashflow": 0.20,
}


def assess_fundamentals(snapshot: dict[str, Any]) -> dict[str, Any]:
    fields = dict(snapshot.get("fields") or snapshot)
    field_sources = dict(snapshot.get("field_sources") or {})
    quality_payload = dict(snapshot.get("quality") or {})
    metric_quality = dict(quality_payload.get("metric_quality") or {})
    metrics: list[dict[str, Any]] = []
    category_values: dict[str, list[tuple[float, float]]] = {
        key: [] for key in _CATEGORY_WEIGHTS
    }
    for spec in _FUNDAMENTAL_SPECS:
        value, source, source_key = _extract(fields, field_sources, spec["aliases"])
        evidence_quality = dict(metric_quality.get(spec["key"]) or {})
        if value is None:
            metrics.append(
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "category": spec["category"],
                    "available": False,
                    "value": None,
                    "display": "--",
                    "score": None,
                    "tone": "unavailable",
                    "source": "",
                    "source_count": 0,
                    "agreement": 0.0,
                    "quality_score": 0,
                }
            )
            continue
        score = round(_clamp(float(spec["score"](value))), 2)
        quality_score = int(evidence_quality.get("quality_score") or 65)
        reliability = _clamp(quality_score / 100, 0.2, 1.0)
        category_values[spec["category"]].append((score, reliability))
        metrics.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "category": spec["category"],
                "available": True,
                "value": round(value, 8),
                "display": _format_metric(value, spec["unit"]),
                "score": score,
                "tone": _metric_tone(score),
                "source": source,
                "source_key": source_key,
                "source_count": int(evidence_quality.get("source_count") or 1),
                "agreement": round(float(evidence_quality.get("agreement") or 0.5), 6),
                "quality_score": quality_score,
            }
        )

    categories: list[dict[str, Any]] = []
    weighted_score = 0.0
    available_weight = 0.0
    for key, weight in _CATEGORY_WEIGHTS.items():
        values = category_values[key]
        reliability_sum = sum(reliability for _value, reliability in values)
        score = (
            sum(value * reliability for value, reliability in values) / reliability_sum
            if reliability_sum
            else None
        )
        if score is not None:
            weighted_score += score * weight
            available_weight += weight
        categories.append(
            {
                "key": key,
                "label": _CATEGORY_LABELS[key],
                "available": score is not None,
                "score": round(score, 2) if score is not None else None,
                "metric_count": len(values),
                "quality": round(100 * reliability_sum / len(values)) if values else 0,
            }
        )

    available_metrics = sum(1 for metric in metrics if metric["available"])
    coverage = available_metrics / len(metrics)
    providers = snapshot.get("providers") or []
    provider_ratio = (
        sum(provider.get("status") == "ok" for provider in providers) / len(providers)
        if providers
        else 1.0
    )
    report_date = str(fields.get("reportDate") or fields.get("报告期") or "")[:10]
    freshness = 1.0
    if report_date:
        try:
            age_days = max(
                0,
                (datetime.now(timezone.utc).date() - datetime.fromisoformat(report_date).date()).days,
            )
            freshness = max(0.55, 1.0 - max(0, age_days - 180) / 1_100)
        except ValueError:
            freshness = 0.85
    data_quality = _finite(quality_payload.get("score"))
    if data_quality is None:
        data_quality = 100 * (0.72 + 0.28 * provider_ratio) * freshness
    confidence = round(
        100 * coverage**0.65 * (0.42 + 0.58 * _clamp(data_quality / 100, 0.0, 1.0))
    )
    score = round(weighted_score / available_weight, 2) if available_weight else 0.0
    if available_metrics == 0:
        label = "数据不足"
    elif score >= 25:
        label = "基本面偏强"
    elif score >= 8:
        label = "基本面略强"
    elif score > -8:
        label = "基本面中性"
    elif score > -25:
        label = "基本面承压"
    else:
        label = "基本面风险"
    company = str(
        fields.get("longName")
        or fields.get("displayName")
        or fields.get("公司名称")
        or snapshot.get("symbol")
        or ""
    )
    return {
        "available": available_metrics > 0,
        "company": company,
        "score": round(_rating_score(score), 2),
        "directional_score": score,
        "rating_score": round(_rating_score(score), 2),
        "label": label,
        "confidence": confidence,
        "coverage": round(coverage, 6),
        "available_metrics": available_metrics,
        "total_metrics": len(metrics),
        "report_date": report_date,
        "data_quality": round(data_quality, 2),
        "data_quality_label": str(quality_payload.get("label") or "来源待核"),
        "cross_source_agreement": round(
            float(quality_payload.get("cross_source_agreement") or 0.0), 6
        ),
        "verified_metric_count": int(quality_payload.get("verified_metric_count") or 0),
        "report_age_days": quality_payload.get("report_age_days"),
        "metrics": metrics,
        "categories": categories,
        "warnings": list(snapshot.get("warnings") or []),
        "method": "source-quality and cross-source agreement weighted scoring; missing values are excluded",
    }


_POSITIVE_ZH = (
    "预增", "扭亏", "增长", "上调", "增持", "回购", "中标", "获批", "突破",
    "创新高", "扩产", "分红", "盈利", "超预期", "签约", "提价", "改善",
)
_NEGATIVE_ZH = (
    "预亏", "预减", "亏损", "下调", "减持", "处罚", "立案", "诉讼", "违约",
    "召回", "退市", "暴跌", "下滑", "终止", "停产", "问询", "风险警示", "爆雷",
)
_UNCERTAIN_ZH = ("可能", "或将", "不确定", "传闻", "尚未", "拟", "预计", "关注函")
_POSITIVE_EN = (
    "beat", "beats", "growth", "upgrade", "upgraded", "buyback", "approval",
    "approved", "record", "profit", "profitable", "dividend", "contract", "surge",
)
_NEGATIVE_EN = (
    "loss", "losses", "downgrade", "downgraded", "investigation", "lawsuit", "default",
    "recall", "decline", "cut", "cuts", "miss", "misses", "warning", "fraud", "bankruptcy",
)
_UNCERTAIN_EN = ("may", "might", "could", "uncertain", "possible", "rumor", "expects")


def _count_terms(text: str, chinese: tuple[str, ...], english: tuple[str, ...]) -> int:
    total = sum(text.count(term) for term in chinese)
    lowered = text.casefold()
    total += sum(len(re.findall(rf"\b{re.escape(term)}\b", lowered)) for term in english)
    return total


def _article_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def assess_news(feed: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    articles = list(feed.get("items") or [])
    scored: list[dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    positive_count = negative_count = neutral_count = 0
    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        positive = _count_terms(text, _POSITIVE_ZH, _POSITIVE_EN)
        negative = _count_terms(text, _NEGATIVE_ZH, _NEGATIVE_EN)
        uncertain = _count_terms(text, _UNCERTAIN_ZH, _UNCERTAIN_EN)
        raw_score = positive - negative - 0.35 * uncertain
        sentiment = 100.0 * math.tanh(raw_score / 2.0)
        published = _article_time(article.get("published_at"))
        age_days = max(0.0, (now - published).total_seconds() / 86_400) if published else 30.0
        recency_weight = 0.5 ** (age_days / 7.0)
        credibility = str(article.get("credibility") or "")
        source_kind = str(article.get("source_kind") or "")
        source_weight = 1.35 if source_kind in {"filing", "exchange"} else 1.0
        if "官方" in credibility or "法定" in credibility or "监管" in credibility:
            source_weight = max(source_weight, 1.25)
        source_count = max(1, int(article.get("verification_count") or 1))
        verification_score = _finite(article.get("verification_score"))
        verification_reliability = (
            _clamp(verification_score / 100, 0.25, 1.0)
            if verification_score is not None
            else min(1.0, 0.30 + 0.14 * source_count)
        )
        weight = (
            max(0.08, recency_weight)
            * source_weight
            * (0.70 + 0.45 * verification_reliability)
        )
        weighted_sum += sentiment * weight
        total_weight += weight
        if sentiment >= 15:
            positive_count += 1
        elif sentiment <= -15:
            negative_count += 1
        else:
            neutral_count += 1
        scored.append(
            {
                "title": str(article.get("title") or ""),
                "url": str(article.get("url") or ""),
                "publisher": str(article.get("publisher") or article.get("provider_label") or ""),
                "published_at": str(article.get("published_at") or ""),
                "score": round(sentiment, 2),
                "age_days": round(age_days, 2),
                "positive_hits": positive,
                "negative_hits": negative,
                "uncertainty_hits": uncertain,
                "verification_count": source_count,
                "verification_status": str(
                    article.get("verification_status") or "single-source"
                ),
                "verification_score": round(100 * verification_reliability, 2),
            }
        )

    score = round(weighted_sum / total_weight, 2) if total_weight else 0.0
    providers = list(feed.get("providers") or [])
    provider_coverage = (
        sum(provider.get("status") == "ok" for provider in providers) / len(providers)
        if providers
        else (1.0 if articles else 0.0)
    )
    official_count = sum(
        str(article.get("source_kind") or "") in {"filing", "exchange"}
        or any(word in str(article.get("credibility") or "") for word in ("官方", "法定", "监管"))
        for article in articles
    )
    official_ratio = official_count / len(articles) if articles else 0.0
    volume_confidence = 1.0 - math.exp(-len(articles) / 5.0)
    lexical_coverage = (
        sum(abs(item["score"]) >= 1 for item in scored) / len(scored) if scored else 0.0
    )
    verification_quality = (
        sum(item["verification_score"] for item in scored) / (100 * len(scored))
        if scored
        else 0.0
    )
    confidence = round(
        100
        * volume_confidence
        * (0.55 + 0.30 * provider_coverage + 0.15 * official_ratio)
        * (0.78 + 0.22 * lexical_coverage)
        * (0.62 + 0.38 * verification_quality)
    )
    if not articles:
        label = "新闻不足"
    elif score >= 20:
        label = "事件偏正面"
    elif score >= 7:
        label = "事件略偏正面"
    elif score > -7:
        label = "事件中性"
    elif score > -20:
        label = "事件略偏负面"
    else:
        label = "事件风险偏高"
    catalysts = sorted(
        (item for item in scored if item["score"] >= 15),
        key=lambda item: (item["score"], -item["age_days"]),
        reverse=True,
    )[:3]
    risks = sorted(
        (item for item in scored if item["score"] <= -15),
        key=lambda item: (item["score"], item["age_days"]),
    )[:3]
    return {
        "available": bool(articles),
        "score": round(_rating_score(score), 2),
        "directional_score": score,
        "rating_score": round(_rating_score(score), 2),
        "label": label,
        "confidence": confidence,
        "article_count": len(articles),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "official_ratio": round(official_ratio, 6),
        "provider_coverage": round(provider_coverage, 6),
        "five_source_verified_count": sum(
            item["verification_status"] == "five-source" for item in scored
        ),
        "mean_source_count": round(
            sum(item["verification_count"] for item in scored) / len(scored), 3
        )
        if scored
        else 0.0,
        "verification_quality": round(verification_quality, 6),
        "catalysts": catalysts,
        "risks": risks,
        "warnings": list(feed.get("warnings") or []),
        "method": "recency and source weighted financial event lexicon",
        "reference": {
            "title": "Loughran-McDonald financial sentiment categories",
            "url": "https://sraf.nd.edu/loughranmcdonald-master-dictionary/",
        },
    }


def build_local_evidence(
    *,
    technical_score: float | int | None,
    technical_confidence: float | int | None,
    quant_validation: dict[str, Any],
    fundamentals: dict[str, Any],
    news: dict[str, Any],
) -> dict[str, Any]:
    base_weights = {
        "technical": 0.35,
        "quant": 0.30,
        "fundamentals": 0.20,
        "news": 0.15,
    }
    technical_value = _finite(technical_score)
    technical_reliability = _clamp((_finite(technical_confidence) or 0.0) / 100, 0.0, 1.0)
    quant_available = bool(quant_validation and quant_validation.get("available"))
    quant_score = 100 * (_finite(quant_validation.get("latest_score")) or 0.0)
    quant_reliability = (
        _clamp((_finite(quant_validation.get("robustness_score")) or 0.0) / 100, 0.0, 1.0)
        if quant_available
        else 0.0
    )
    fundamental_available = bool(fundamentals.get("available"))
    news_available = bool(news.get("available"))
    component_data = (
        (
            "technical",
            "技术因子",
            technical_value or 0.0,
            technical_reliability,
            technical_value is not None,
            "27 项价格、波动、流动性与尾部风险因子",
        ),
        (
            "quant",
            "样本外验证",
            quant_score,
            quant_reliability,
            quant_available,
            f"{len(quant_validation.get('folds') or [])} 折 · {quant_validation.get('verdict', '不可用')}",
        ),
        (
            "fundamentals",
            "基本面",
            _finite(
                fundamentals.get("directional_score", fundamentals.get("score"))
            )
            or 0.0,
            (_finite(fundamentals.get("confidence")) or 0.0) / 100,
            fundamental_available,
            f"{fundamentals.get('available_metrics', 0)} / {fundamentals.get('total_metrics', 0)} 项",
        ),
        (
            "news",
            "新闻事件",
            _finite(news.get("directional_score", news.get("score"))) or 0.0,
            (_finite(news.get("confidence")) or 0.0) / 100,
            news_available,
            f"{news.get('article_count', 0)} 条 · 官方占比 {(_finite(news.get('official_ratio')) or 0.0):.0%}",
        ),
    )
    components: list[dict[str, Any]] = []
    evidence_weights: dict[str, float] = {}
    denominator = 0.0
    numerator = 0.0
    coverage = 0.0
    confidence_mass = 0.0
    for key, label, score, reliability, available, detail in component_data:
        reliability = _clamp(reliability, 0.0, 1.0)
        base_weight = base_weights[key]
        evidence_weight = base_weight * reliability if available else 0.0
        denominator += evidence_weight
        numerator += score * evidence_weight
        evidence_weights[key] = evidence_weight
        if available:
            coverage += base_weight
        confidence_mass += evidence_weight
        components.append(
            {
                "key": key,
                "label": label,
                "available": available,
                "score": round(_rating_score(score), 2) if available else None,
                "directional_score": round(_clamp(score), 2) if available else None,
                "direction": _direction_phrase(score) if available else "数据不足",
                "confidence": round(reliability * 100) if available else 0,
                "base_weight": base_weight,
                "effective_weight": 0.0,
                "detail": detail,
            }
        )
    raw_directional_score = _clamp(numerator / denominator) if denominator else 0.0
    for component in components:
        key = component["key"]
        if component["available"] and denominator:
            component["effective_weight"] = round(evidence_weights[key] / denominator, 6)
    weighted_deviation = sum(
        evidence_weights[component["key"]]
        / denominator
        * abs(float(component["directional_score"]) - raw_directional_score)
        for component in components
        if component["available"] and denominator
    )
    agreement = _clamp(1.0 - weighted_deviation / 100.0, 0.0, 1.0)
    overall_reliability = _clamp(
        confidence_mass * (0.65 + 0.35 * agreement), 0.0, 1.0
    )
    calibration_factor = 0.35 + 0.65 * overall_reliability
    directional_score = _clamp(raw_directional_score * calibration_factor)
    score = _rating_score(directional_score)
    confidence = round(100 * overall_reliability)
    if directional_score >= 25:
        label = "多头证据占优"
    elif directional_score >= 8:
        label = "证据略偏多"
    elif directional_score > -8:
        label = "证据中性"
    elif directional_score > -25:
        label = "证据略偏空"
    else:
        label = "空头风险占优"
    available_scores = [
        component["directional_score"]
        for component in components
        if component["available"]
    ]
    conflicts: list[str] = []
    if available_scores and max(available_scores) - min(available_scores) >= 60:
        high = max(
            (item for item in components if item["available"]),
            key=lambda item: item["directional_score"],
        )
        low = min(
            (item for item in components if item["available"]),
            key=lambda item: item["directional_score"],
        )
        conflicts.append(f"{high['label']}与{low['label']}方向分歧较大")
    missing = [component["label"] for component in components if not component["available"]]
    summary = (
        f"本地研判评分 {score:.1f}/100，方向{_direction_phrase(directional_score)}"
        f"（强度 {abs(directional_score):.1f}），可信度 {confidence}%"
    )
    if conflicts:
        summary += f"；{conflicts[0]}。"
    elif missing:
        summary += f"；{ '、'.join(missing) }缺失，已从权重中剔除。"
    else:
        summary += "；四类证据均已纳入并按各自可靠度加权。"
    return {
        "score": round(score, 2),
        "directional_score": round(directional_score, 2),
        "raw_directional_score": round(raw_directional_score, 2),
        "signal_strength": round(abs(directional_score), 2),
        "label": label,
        "confidence": confidence,
        "agreement": round(agreement, 6),
        "calibration_factor": round(calibration_factor, 6),
        "coverage": round(coverage, 6),
        "components": components,
        "conflicts": conflicts,
        "missing_components": missing,
        "summary": summary,
        "method": {
            "weights": base_weights,
            "score_scale": "0-100 rating; 50 is neutral",
            "direction_scale": "-100 bearish to +100 bullish",
            "missing_policy": "exclude and renormalize",
            "confidence_policy": "source reliability multiplied by cross-component agreement",
            "calibration_policy": "direction shrinks toward neutral when evidence is weak or conflicting",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
