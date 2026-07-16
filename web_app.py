from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
import hmac
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from typing import Any, Literal, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if not SRC_ROOT.exists():
    SRC_ROOT = BUNDLE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from quant_starter.agent_workflow import (
    ProgressEvent,
    RavenWatchAgentsWorkflow,
    WorkflowCancelled,
    WorkflowConfig,
)
from quant_starter.data import (
    _fetch_tencent_ohlcv_direct,
    fetch_a_share_ohlcv,
    fetch_global_ohlcv,
    fetch_hk_ohlcv,
    fetch_nasdaq_ohlcv,
    generate_demo_ohlcv,
    provider_display_name,
    provider_source_details,
    summarize_error,
)
from quant_starter.factors import (
    FACTOR_WEIGHTS,
    analyze_composite,
    build_factor_history,
    factor_reference_catalog,
)
from quant_starter.llm_client import (
    LLM_PROVIDER_PROFILES,
    LLMRequestError,
    LLMSettings,
    OpenAICompatibleClient,
    provider_profile,
)
from quant_starter.local_evidence import (
    assess_fundamentals,
    assess_news,
    build_local_evidence,
)
from quant_starter.news import fetch_online_news
from quant_starter.global_market import search_msn_instruments
from quant_starter.instrument_catalog import catalog_service
from quant_starter.realtime import (
    MarketSnapshot,
    RealtimeMarketData,
    fetch_realtime_market,
    intraday_records,
)
from quant_starter.research_data import (
    ResearchContext,
    calculate_technical_snapshot,
    fetch_fundamental_snapshot,
    fetch_research_evidence,
)
from quant_starter.symbols import (
    MARKET_DEFINITIONS,
    display_for_symbol,
    market_definition,
    preset_for_symbol,
    presets_for_source,
    resolve_stock_choice,
    search_local_presets,
)
from quant_starter.walk_forward import (
    WalkForwardConfig,
    adaptive_walk_forward_config,
    walk_forward_validate,
)


WEB_ROOT = BUNDLE_ROOT / "web"
T = TypeVar("T")
AnalystId = Literal["market", "sentiment", "news", "fundamentals"]
ProviderId = Literal["openai", "deepseek", "qwen", "ollama"]


@dataclass(frozen=True)
class AgentResearchOptions:
    mode: str = "offline"
    provider: str = "deepseek"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.2
    timeout_seconds: int = 60
    thinking_mode: str = "enabled"
    reasoning_effort: str = "high"
    max_tokens: int = 2_000
    fallback_to_offline: bool = True
    fetch_details: bool = False
    selected_analysts: tuple[str, ...] = (
        "market",
        "sentiment",
        "news",
        "fundamentals",
    )
    debate_rounds: int = 1
    risk_rounds: int = 1


def _provider_environment_value(provider: str, suffix: str) -> str:
    specific = os.getenv(f"RAVENWATCHAGENTS_{provider.upper()}_{suffix}", "").strip()
    configured_provider = os.getenv("RAVENWATCHAGENTS_LLM_PROVIDER", "").strip().lower()
    generic = os.getenv(f"RAVENWATCHAGENTS_LLM_{suffix}", "").strip()
    return specific or (generic if configured_provider == provider else "")


def _resolve_llm_runtime(
    options: AgentResearchOptions,
) -> tuple[OpenAICompatibleClient | None, dict[str, Any]]:
    if options.mode == "offline":
        return None, {"mode": "offline", "provider": "rules", "model": "auditable-rules"}

    profile = provider_profile(options.provider)
    model = (
        options.model.strip()
        or _provider_environment_value(profile.provider_id, "MODEL")
        or profile.default_model
    )
    if not model:
        raise ValueError("在线分析必须填写模型 ID，或在服务端配置模型环境变量。")
    api_key = options.api_key.strip()
    if not api_key and profile.api_key_env:
        api_key = os.getenv(profile.api_key_env, "").strip()
    if not api_key:
        api_key = _provider_environment_value(profile.provider_id, "API_KEY")
    if profile.requires_api_key and not api_key:
        raise ValueError(f"{profile.label} 在线分析需要 API Key。")

    base_url = _provider_environment_value(profile.provider_id, "BASE_URL") or profile.base_url
    settings = LLMSettings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=options.temperature,
        timeout_seconds=options.timeout_seconds,
        provider_id=profile.provider_id,
        thinking_mode=options.thinking_mode,
        reasoning_effort=options.reasoning_effort,
        max_tokens=options.max_tokens,
    )
    client = OpenAICompatibleClient(settings)
    runtime: dict[str, Any] = {
        "mode": "online",
        "provider": profile.provider_id,
        "provider_label": profile.label,
        "model": model,
        "base_url": base_url,
        "server_key_used": not bool(options.api_key.strip()) and bool(api_key),
    }
    if profile.supports_thinking:
        runtime["thinking"] = {
            "type": options.thinking_mode,
            "reasoning_effort": (
                options.reasoning_effort
                if options.thinking_mode == "enabled"
                else None
            ),
        }
    return client, runtime


def _workflow_config(options: AgentResearchOptions) -> WorkflowConfig:
    config = WorkflowConfig(
        mode=options.mode,
        selected_analysts=options.selected_analysts,
        debate_rounds=options.debate_rounds,
        risk_rounds=options.risk_rounds,
        fallback_to_offline=options.fallback_to_offline,
    )
    config.validate()
    return config


def _validate_agent_options(
    options: AgentResearchOptions,
) -> tuple[WorkflowConfig, OpenAICompatibleClient | None, dict[str, Any]]:
    config = _workflow_config(options)
    client, runtime = _resolve_llm_runtime(options)
    return config, client, runtime


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CacheEntry:
    value: Any
    expires_at: float
    stale_until: float


class TTLCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], CacheEntry] = {}
        self._lock = threading.RLock()

    def get_or_load(
        self,
        key: tuple[Any, ...],
        loader: Callable[[], T],
        *,
        ttl_seconds: float,
        stale_seconds: float = 0,
    ) -> T:
        now = time.monotonic()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and now < cached.expires_at:
                return cached.value
        try:
            value = loader()
        except Exception:
            with self._lock:
                cached = self._entries.get(key)
                if cached is not None and now < cached.stale_until:
                    return cached.value
            raise
        with self._lock:
            self._entries[key] = CacheEntry(
                value=value,
                expires_at=now + ttl_seconds,
                stale_until=now + ttl_seconds + stale_seconds,
            )
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


@dataclass
class ResearchJob:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.updated_at = _utc_now_iso()
            self.events.append(
                {
                    "id": len(self.events) + 1,
                    "event": event,
                    "data": data,
                }
            )

    def set_status(self, status: str, *, error: str = "") -> None:
        with self.lock:
            self.status = status
            self.error = error
            self.updated_at = _utc_now_iso()

    def events_after(self, cursor: int) -> tuple[list[dict[str, Any]], str]:
        with self.lock:
            return [dict(item) for item in self.events if item["id"] > cursor], self.status

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "job_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "event_count": len(self.events),
                "error": self.error or None,
            }
            if self.status == "completed":
                payload["result"] = self.result
            return payload


class ResearchJobManager:
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

    def __init__(self, *, max_active: int = 2, retention_seconds: int = 3_600) -> None:
        self.max_active = max_active
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, ResearchJob] = {}
        self._created_at: dict[str, float] = {}
        self._lock = threading.RLock()

    def create(
        self,
        runner: Callable[
            [Callable[[ProgressEvent], None], threading.Event], dict[str, Any]
        ],
    ) -> ResearchJob:
        with self._lock:
            self._cleanup_locked()
            active = sum(
                job.status not in self.TERMINAL_STATUSES for job in self._jobs.values()
            )
            if active >= self.max_active:
                raise ValueError(f"最多同时运行 {self.max_active} 个智能体任务。")
            job_id = secrets.token_urlsafe(18)
            job = ResearchJob(job_id=job_id)
            self._jobs[job_id] = job
            self._created_at[job_id] = time.monotonic()

        worker = threading.Thread(
            target=self._run,
            args=(job, runner),
            name=f"research-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return job

    def get(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ResearchJob | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.status not in self.TERMINAL_STATUSES:
            job.cancel_event.set()
            job.set_status("cancelling")
            job.publish("state", {"status": "cancelling", "message": "正在取消研判"})
        return job

    def _run(
        self,
        job: ResearchJob,
        runner: Callable[
            [Callable[[ProgressEvent], None], threading.Event], dict[str, Any]
        ],
    ) -> None:
        job.set_status("running")
        job.publish("state", {"status": "running", "message": "智能体任务已启动"})

        def progress(event: ProgressEvent) -> None:
            job.publish("progress", asdict(event))

        try:
            result = runner(progress, job.cancel_event)
        except WorkflowCancelled as exc:
            job.set_status("cancelled")
            job.publish("state", {"status": "cancelled", "message": str(exc)})
        except Exception as exc:
            detail = summarize_error(exc, 500)
            job.set_status("failed", error=detail)
            job.publish("state", {"status": "failed", "message": detail})
        else:
            with job.lock:
                job.result = result
            job.set_status("completed")
            job.publish(
                "state",
                {
                    "status": "completed",
                    "message": "全部智能体已完成",
                    "agent_summary": result.get("agent_summary", {}),
                },
            )

    def _cleanup_locked(self) -> None:
        cutoff = time.monotonic() - self.retention_seconds
        expired = [
            job_id
            for job_id, created in self._created_at.items()
            if created < cutoff
            and self._jobs[job_id].status in self.TERMINAL_STATUSES
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._created_at.pop(job_id, None)


PERIOD_DAYS = {
    "3m": 120,
    "6m": 220,
    "1y": 420,
    "3y": 1_100,
    "5y": 1_900,
}


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    aliases = {
        "a": "a-share",
        "ashare": "a-share",
        "us": "nasdaq",
        "usa": "nasdaq",
        "hong-kong": "hk",
        "world": "global",
        "international": "global",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"a-share", "nasdaq", "hk", "global", "demo"}:
        raise ValueError("市场必须是 a-share、nasdaq、hk、global 或 demo。")
    return normalized


def _normalize_symbol(market: str, symbol: str) -> str:
    return resolve_stock_choice(symbol, market)


def _date_window(period: str) -> tuple[str, str]:
    if period not in PERIOD_DAYS:
        raise ValueError("K 线周期必须是 3m、6m、1y、3y 或 5y。")
    end = date.today()
    start = end - timedelta(days=PERIOD_DAYS[period])
    return start.isoformat(), end.isoformat()


def _serialize_bars(bars: pd.DataFrame, limit: int = 1_500) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for timestamp, row in bars.tail(limit).iterrows():
        records.append(
            {
                "date": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 6),
                "high": round(float(row["High"]), 6),
                "low": round(float(row["Low"]), 6),
                "close": round(float(row["Close"]), 6),
                "volume": round(float(row["Volume"]), 2),
            }
        )
    return records


def _merge_live_candle(
    bars: pd.DataFrame,
    live: RealtimeMarketData | None,
) -> pd.DataFrame:
    if live is None or live.intraday.empty:
        return bars
    intraday = live.intraday
    timestamp = pd.Timestamp(intraday.index[-1])
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    candle_date = timestamp.normalize()
    live_volume = float(intraday["CumulativeVolume"].iloc[-1])
    if live.snapshot.market == "a-share":
        live_volume /= 100
    row = pd.DataFrame(
        {
            "Open": [float(intraday["Price"].iloc[0])],
            "High": [float(intraday["Price"].max())],
            "Low": [float(intraday["Price"].min())],
            "Close": [float(intraday["Price"].iloc[-1])],
            "Volume": [live_volume],
        },
        index=pd.DatetimeIndex([candle_date]),
    )
    merged = bars.copy()
    merged = merged.loc[merged.index.normalize() != candle_date]
    merged = pd.concat([merged, row]).sort_index()
    merged.attrs.update(bars.attrs)
    merged.attrs["live_candle"] = True
    return merged


def _fallback_snapshot(market: str, symbol: str, bars: pd.DataFrame) -> MarketSnapshot:
    latest = bars.iloc[-1]
    previous = bars.iloc[-2] if len(bars) > 1 else latest
    price = float(latest["Close"])
    previous_close = float(previous["Close"])
    change = price - previous_close
    instrument = bars.attrs.get("instrument") or {}
    preset = preset_for_symbol(market, symbol)
    currency = str(instrument.get("currency") or (preset.currency if preset else ""))
    if not currency:
        currency = "CNY" if market == "a-share" else "HKD" if market == "hk" else "USD"
    return MarketSnapshot(
        market=market,
        symbol=symbol,
        name=display_for_symbol(market, symbol).split(" · ")[0],
        currency=currency,
        price=price,
        previous_close=previous_close,
        change=change,
        change_pct=change / previous_close if previous_close else 0.0,
        open=float(latest["Open"]),
        high=float(latest["High"]),
        low=float(latest["Low"]),
        volume=float(latest["Volume"]),
        amount=0.0,
        timestamp=pd.Timestamp(bars.index[-1]).isoformat(),
        session_status="historical",
        provider=str(bars.attrs.get("provider", "historical")),
        exchange=str(instrument.get("exchange") or (preset.exchange if preset else "")),
        country=str(instrument.get("country") or (preset.country if preset else "")),
        asset_type=str(instrument.get("asset_type") or (preset.asset_type if preset else "stock")),
        asset_type_label=str(instrument.get("asset_type_label") or "证券"),
        timezone=str(instrument.get("timezone") or ""),
        source_url=str(bars.attrs.get("provider_url") or ""),
    )


def _data_quality_payload(
    bars: pd.DataFrame,
    *,
    period: str,
    adjustment: str,
) -> dict[str, Any]:
    required = ["Open", "High", "Low", "Close", "Volume"]
    available = [column for column in required if column in bars.columns]
    expected_cells = max(1, len(bars) * len(required))
    missing_cells = int(bars.reindex(columns=required).isna().sum().sum())
    completeness = max(0.0, 1.0 - missing_cells / expected_cells)
    duplicate_dates = int(pd.Index(bars.index).duplicated().sum())
    invalid_rows = 0
    if {"Open", "High", "Low", "Close"}.issubset(bars.columns):
        invalid = (
            (bars[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
            | (bars["High"] < bars[["Open", "Close", "Low"]].max(axis=1))
            | (bars["Low"] > bars[["Open", "Close", "High"]].min(axis=1))
        )
        invalid_rows = int(invalid.sum())
    rows = len(bars)
    if rows >= 252:
        sample_status = "充分"
    elif rows >= 120:
        sample_status = "可用"
    elif rows >= 60:
        sample_status = "有限"
    else:
        sample_status = "不足"
    penalties = missing_cells + duplicate_dates * 5 + invalid_rows * 5
    score = max(0, round(100 - penalties / expected_cells * 100))
    start = pd.Timestamp(bars.index[0]).date().isoformat() if rows else ""
    end = pd.Timestamp(bars.index[-1]).date().isoformat() if rows else ""
    return {
        "score": score,
        "status": sample_status,
        "rows": rows,
        "start": start,
        "end": end,
        "completeness": completeness,
        "missing_cells": missing_cells,
        "duplicate_dates": duplicate_dates,
        "invalid_rows": invalid_rows,
        "available_fields": available,
        "period": period,
        "adjustment": adjustment,
    }


def _walk_forward_payload(
    bars: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    resolved = config or adaptive_walk_forward_config(len(bars))
    minimum_rows = resolved.train_rows + resolved.test_rows * 2
    if len(bars) < minimum_rows:
        return {
            "available": False,
            "status": "insufficient_history",
            "rows": len(bars),
            "required_rows": minimum_rows,
            "message": f"滚动样本外验证至少需要 {minimum_rows} 根日线，当前只有 {len(bars)} 根。",
        }
    try:
        return walk_forward_validate(bars, config=resolved).to_dict()
    except ValueError as exc:
        return {
            "available": False,
            "status": "invalid_history",
            "rows": len(bars),
            "required_rows": minimum_rows,
            "message": str(exc),
        }


class MarketService:
    def __init__(self) -> None:
        self.cache = TTLCache()

    def get_live(self, market: str, symbol: str, *, force: bool = False) -> RealtimeMarketData:
        if market == "demo":
            raise ValueError("模拟市场不提供网络分时行情。")
        if force:
            key = ("live-force", market, symbol, time.monotonic_ns())
        else:
            key = ("live", market, symbol)
        return self.cache.get_or_load(
            key,
            lambda: fetch_realtime_market(market, symbol),
            ttl_seconds=8,
            stale_seconds=600,
        )

    def get_bars(
        self,
        market: str,
        symbol: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        key = ("bars", market, symbol, period, adjust)

        def load() -> pd.DataFrame:
            start, end = _date_window(period)
            if market == "a-share":
                try:
                    return _fetch_tencent_ohlcv_direct(
                        symbol,
                        pd.Timestamp(start).strftime("%Y%m%d"),
                        pd.Timestamp(end).strftime("%Y%m%d"),
                        adjust,
                    )
                except Exception:
                    return fetch_a_share_ohlcv(symbol, start, end, adjust)
            if market == "nasdaq":
                return fetch_nasdaq_ohlcv(symbol, start, end, auto_adjust=True)
            if market == "hk":
                return fetch_hk_ohlcv(symbol, start, end, adjust)
            if market == "global":
                return fetch_global_ohlcv(symbol, start, end, auto_adjust=True)
            return generate_demo_ohlcv(symbol).loc[start:end]

        return self.cache.get_or_load(
            key,
            load,
            ttl_seconds=300,
            stale_seconds=3_600,
        )

    def get_news(
        self,
        market: str,
        symbol: str,
        *,
        limit: int = 12,
        force: bool = False,
    ) -> dict[str, Any]:
        if force:
            key = ("news-force", market, symbol, limit, time.monotonic_ns())
        else:
            key = ("news", market, symbol, limit)
        company_name = display_for_symbol(market, symbol).split(" · ")[0]
        feed = self.cache.get_or_load(
            key,
            lambda: fetch_online_news(
                market,
                symbol,
                company_name=company_name,
                limit=limit,
                timeout_seconds=12,
            ),
            ttl_seconds=90,
            stale_seconds=1_800,
        )
        return feed.to_dict()

    def get_fundamentals(
        self,
        market: str,
        symbol: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if force:
            key = ("fundamentals-force", market, symbol, time.monotonic_ns())
        else:
            key = ("fundamentals", market, symbol)
        snapshot = self.cache.get_or_load(
            key,
            lambda: fetch_fundamental_snapshot(
                market,
                symbol,
                timeout_seconds=12,
            ),
            ttl_seconds=900,
            stale_seconds=21_600,
        )
        return snapshot.to_dict()

    def dashboard_payload(
        self,
        market: str,
        symbol: str,
        period: str,
        adjust: str,
    ) -> dict[str, Any]:
        bars = self.get_bars(market, symbol, period, adjust)
        live: RealtimeMarketData | None = None
        warnings: list[str] = []
        if market != "demo":
            try:
                live = self.get_live(market, symbol)
            except Exception as exc:
                warnings.append("实时分时暂不可用：" + summarize_error(exc))
        snapshot = live.snapshot if live is not None else _fallback_snapshot(market, symbol, bars)
        display_bars = _merge_live_candle(bars, live)
        analysis = analyze_composite(bars)
        return {
            "market": market,
            "symbol": symbol,
            "snapshot": snapshot.to_dict(),
            "candles": _serialize_bars(display_bars),
            "intraday": intraday_records(live.intraday) if live is not None else [],
            "analysis": analysis.to_dict(),
            "data_quality": _data_quality_payload(
                bars,
                period=period,
                adjustment=adjust,
            ),
            "quant_validation": _walk_forward_payload(bars),
            "factor_history": build_factor_history(bars, periods=30),
            "provider": {
                "daily": str(bars.attrs.get("provider", "unknown")),
                "daily_label": provider_display_name(bars.attrs.get("provider")),
                "daily_url": provider_source_details(
                    bars.attrs.get("provider"),
                    str(bars.attrs.get("provider_url", "")),
                )["url"],
                "realtime": snapshot.provider,
                "realtime_label": provider_display_name(snapshot.provider),
                "realtime_url": snapshot.source_url or provider_source_details(snapshot.provider)["url"],
            },
            "warnings": warnings,
            "refreshed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        }

    def quant_validation_payload(
        self,
        market: str,
        symbol: str,
        period: str,
        adjust: str,
        config: WalkForwardConfig | None = None,
    ) -> dict[str, Any]:
        bars = self.get_bars(market, symbol, period, adjust)
        return {
            "market": market,
            "symbol": symbol,
            "period": period,
            "adjustment": adjust,
            "provider": str(bars.attrs.get("provider", "unknown")),
            "validation": _walk_forward_payload(bars, config),
        }

    def evidence_payload(
        self,
        market: str,
        symbol: str,
        period: str,
        adjust: str,
        *,
        config: WalkForwardConfig | None = None,
        limit: int = 12,
        force: bool = False,
    ) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="evidence") as pool:
            bars_future = pool.submit(self.get_bars, market, symbol, period, adjust)
            fundamentals_future = pool.submit(
                self.get_fundamentals,
                market,
                symbol,
                force=force,
            )
            news_future = pool.submit(
                self.get_news,
                market,
                symbol,
                limit=limit,
                force=force,
            )
            bars = bars_future.result()
            fundamentals_snapshot = fundamentals_future.result()
            news_feed = news_future.result()

        analysis = analyze_composite(bars).to_dict()
        quant_validation = _walk_forward_payload(bars, config)
        fundamental_assessment = assess_fundamentals(fundamentals_snapshot)
        news_assessment = assess_news(news_feed)
        local_evidence = build_local_evidence(
            technical_score=analysis.get("score"),
            technical_confidence=analysis.get("confidence"),
            quant_validation=quant_validation,
            fundamentals=fundamental_assessment,
            news=news_assessment,
        )
        return {
            "market": market,
            "symbol": symbol,
            "period": period,
            "adjustment": adjust,
            "fundamentals": fundamentals_snapshot,
            "fundamental_assessment": fundamental_assessment,
            "news": news_feed,
            "news_assessment": news_assessment,
            "local_evidence": local_evidence,
            "quant_validation": quant_validation,
            "refreshed_at": _utc_now_iso(),
        }

    def live_payload(self, market: str, symbol: str) -> dict[str, Any]:
        live = self.get_live(market, symbol)
        return {
            "snapshot": live.snapshot.to_dict(),
            "intraday": intraday_records(live.intraday),
            "refreshed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        }

    def run_agents(
        self,
        market: str,
        symbol: str,
        *,
        options: AgentResearchOptions | None = None,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        options = options or AgentResearchOptions()
        workflow_config, llm_client, llm_runtime = _validate_agent_options(options)

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise WorkflowCancelled("用户已取消本次投研。")

        check_cancelled()
        bars = self.get_bars(market, symbol, "1y", "qfq")
        check_cancelled()
        technical = calculate_technical_snapshot(bars)
        technical["data_provider"] = str(bars.attrs.get("provider", "unknown"))
        technical["data_provider_label"] = provider_display_name(
            bars.attrs.get("provider")
        )
        fundamentals: dict[str, Any] = {}
        news: list[dict[str, str]] = []
        warnings: list[str] = []
        if options.fetch_details:
            fundamentals, news, evidence_warnings = fetch_research_evidence(
                market,
                symbol,
                timeout_seconds=min(30, options.timeout_seconds),
            )
            warnings.extend(evidence_warnings)
            try:
                online_feed = self.get_news(market, symbol, limit=12)
                online_news = [
                    {
                        "title": item["title"],
                        "publisher": item["publisher"],
                        "published": item["published_at"],
                        "url": item["url"],
                    }
                    for item in online_feed.get("items", [])
                ]
                if online_news:
                    news = online_news
                    warnings = [
                        warning.replace("基本面/新闻暂不可用", "基本面暂不可用")
                        for warning in warnings
                        if not warning.startswith("新闻数据为空")
                    ]
                warnings.extend(online_feed.get("warnings", []))
            except Exception as exc:
                warnings.append("在线新闻聚合暂不可用：" + summarize_error(exc))
            check_cancelled()
        else:
            warnings.append("本次快速研判未启用基本面和新闻补充。")
        context = ResearchContext(
            symbol=symbol,
            market=market,
            analysis_date=str(pd.Timestamp(bars.index[-1]).date()),
            bars=bars,
            technical=technical,
            fundamentals=fundamentals,
            news=news,
            warnings=warnings,
        )
        workflow = RavenWatchAgentsWorkflow(workflow_config, llm_client)
        result = workflow.run(
            context,
            progress=progress,
            cancel_event=cancel_event,
        )
        summary = result.summary_dict()
        if llm_client is not None:
            llm_runtime["usage"] = llm_client.usage_summary()
        status_counts: dict[str, int] = {}
        for item in result.agent_runs:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        return {
            "mode": result.mode,
            "llm": llm_runtime,
            "decision": summary["decision"],
            "reports": result.reports,
            "report_titles": result.report_titles,
            "agent_runs": summary["agent_runs"],
            "agent_summary": {
                "total": len(result.agent_runs),
                "status_counts": status_counts,
                "all_completed": all(
                    item.status in {"completed", "fallback"}
                    for item in result.agent_runs
                ),
            },
            "evidence": {
                "fundamental_fields": len(fundamentals),
                "news_items": len(news),
            },
            "warnings": result.warnings,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
        }


service = MarketService()
research_jobs = ResearchJobManager()


class AgentResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(pattern=r"^[A-Za-z-]+$")
    symbol: str = Field(min_length=1, max_length=32)
    mode: Literal["offline", "online"] = "offline"
    provider: ProviderId = "deepseek"
    model: str = Field(default="", max_length=160)
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: int = Field(default=60, ge=10, le=300)
    thinking_mode: Literal["enabled", "disabled"] = "enabled"
    reasoning_effort: Literal["high", "max"] = "high"
    fallback_to_offline: bool = True
    fetch_details: bool = False
    selected_analysts: tuple[AnalystId, ...] = (
        "market",
        "sentiment",
        "news",
        "fundamentals",
    )
    debate_rounds: int = Field(default=1, ge=1, le=3)
    risk_rounds: int = Field(default=1, ge=1, le=3)

    def options(self) -> AgentResearchOptions:
        selected = tuple(dict.fromkeys(self.selected_analysts))
        return AgentResearchOptions(
            mode=self.mode,
            provider=self.provider,
            model=self.model.strip(),
            api_key=self.api_key.get_secret_value() if self.api_key else "",
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort,
            fallback_to_offline=self.fallback_to_offline,
            fetch_details=self.fetch_details,
            selected_analysts=selected,
            debate_rounds=self.debate_rounds,
            risk_rounds=self.risk_rounds,
        )


class ProviderConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderId = "deepseek"
    model: str = Field(default="", max_length=160)
    api_key: SecretStr | None = None
    timeout_seconds: int = Field(default=30, ge=10, le=60)
    thinking_mode: Literal["enabled", "disabled"] = "enabled"
    reasoning_effort: Literal["high", "max"] = "high"

    def options(self) -> AgentResearchOptions:
        return AgentResearchOptions(
            mode="online",
            provider=self.provider,
            model=self.model.strip(),
            api_key=self.api_key.get_secret_value() if self.api_key else "",
            temperature=0,
            timeout_seconds=self.timeout_seconds,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort,
            max_tokens=64,
        )


class QuantValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(default="a-share", pattern=r"^[A-Za-z-]+$")
    symbol: str = Field(default="300750", min_length=1, max_length=32)
    period: str = Field(default="1y", pattern=r"^(3m|6m|1y|3y|5y)$")
    adjust: Literal["none", "qfq", "hfq"] = "qfq"
    train_rows: int = Field(default=0, ge=0, le=756)
    test_rows: int = Field(default=0, ge=0, le=126)
    commission_bps: float = Field(default=3.0, ge=0, le=100)
    slippage_bps: float = Field(default=2.0, ge=0, le=100)
    stress_multiplier: float = Field(default=2.0, ge=1, le=5)
    max_position_percent: float = Field(default=100.0, ge=5, le=100)
    bootstrap_horizon: int = Field(default=63, ge=5, le=252)
    bootstrap_simulations: int = Field(default=1_000, ge=100, le=5_000)
    bootstrap_block_size: int = Field(default=5, ge=1, le=63)
    random_seed: int = Field(default=7, ge=0, le=2_147_483_647)

    def walk_forward_config(self, row_count: int) -> WalkForwardConfig:
        adaptive = adaptive_walk_forward_config(row_count)
        train_rows = self.train_rows or adaptive.train_rows
        test_rows = self.test_rows or adaptive.test_rows
        return WalkForwardConfig(
            train_rows=train_rows,
            test_rows=test_rows,
            minimum_test_rows=min(21, test_rows),
            commission_rate=self.commission_bps / 10_000,
            slippage_rate=self.slippage_bps / 10_000,
            stress_multiplier=self.stress_multiplier,
            bootstrap_horizon=self.bootstrap_horizon,
            bootstrap_simulations=self.bootstrap_simulations,
            bootstrap_block_size=self.bootstrap_block_size,
            random_seed=self.random_seed,
            max_position=self.max_position_percent / 100,
        )


class EvidenceRequest(QuantValidationRequest):
    limit: int = Field(default=12, ge=1, le=30)
    force: bool = False


app = FastAPI(
    title="Raven Watch Agents Pro API",
    version="1.7.0",
    description="四市场实时行情、参数化样本外验证、新闻基本面与 DeepSeek V4 多智能体在线研判服务。",
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    desktop_token = os.getenv("RAVENWATCHAGENTS_DESKTOP_TOKEN", "")
    if desktop_token and request.url.path.startswith("/api/"):
        supplied_token = (
            request.headers.get("X-Desktop-Token")
            or request.query_params.get("desktop_token")
            or ""
        )
        if not hmac.compare_digest(desktop_token, supplied_token):
            return JSONResponse(status_code=403, content={"detail": "桌面会话令牌无效。"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
        "font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "Raven Watch Agents Pro",
        "version": app.version,
        "markets": ["a-share", "nasdaq", "hk", "global"],
        "realtime": True,
        "online_news": True,
        "instrument_search": True,
        "instrument_catalog": True,
        "walk_forward_validation": True,
        "configurable_quant": True,
        "local_evidence_scoring": True,
        "online_fundamentals": True,
        "deepseek_v4_online": True,
        "factor_count": len(FACTOR_WEIGHTS),
    }


def _preset_payload(item: Any) -> dict[str, str]:
    return {
        "symbol": item.symbol,
        "name": item.name,
        "label": item.label,
        "market": item.source,
        "asset_type": item.asset_type,
        "asset_type_label": {
            "stock": "股票",
            "etf": "ETF",
            "fund": "基金",
            "index": "指数",
            "reit": "REIT",
        }.get(item.asset_type, item.asset_type.upper()),
        "exchange": item.exchange,
        "country": item.country,
        "currency": item.currency,
        "category": item.category,
        "source_url": "",
    }


@app.get("/api/markets")
def markets() -> dict[str, Any]:
    return {
        "markets": [
            {
                **market_definition(market_id),
                "preset_count": len(presets_for_source(market_id)),
                "catalog_count": catalog_service.summary(market_id)["count"],
                "asset_types": sorted(
                    {item.asset_type for item in presets_for_source(market_id)}
                ),
            }
            for market_id in ("a-share", "nasdaq", "hk", "global")
        ]
    }


@app.get("/api/instruments/catalog")
async def instrument_catalog(
    market: str = Query("a-share"),
    q: str = Query("", max_length=80),
    asset_type: str = Query("all", pattern=r"^(all|stock|etf|fund|index|reit)$"),
    category: str = Query("all", max_length=80),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=100),
    refresh: bool = Query(False),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    try:
        return await asyncio.to_thread(
            catalog_service.query,
            market=normalized_market,
            q=q,
            asset_type=asset_type,
            category=category,
            page=page,
            page_size=page_size,
            refresh=refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/symbols")
def symbols(
    market: str = Query("a-share"),
    asset_type: str = Query("all", pattern=r"^(all|stock|etf|fund|index|reit)$"),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    return {
        "market": normalized_market,
        "definition": market_definition(normalized_market),
        "symbols": [
            _preset_payload(item)
            for item in presets_for_source(normalized_market, asset_type)
        ],
    }


@app.get("/api/instruments/search")
async def search_instruments(
    q: str = Query(..., min_length=1, max_length=80),
    market: str = Query("global"),
    asset_type: str = Query("all", pattern=r"^(all|stock|etf|fund|index|reit)$"),
    limit: int = Query(12, ge=1, le=30),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    local = [
        _preset_payload(item)
        for item in search_local_presets(q, normalized_market, asset_type, limit)
    ]
    warnings: list[str] = []
    online: list[dict[str, str]] = []
    try:
        remote = await asyncio.to_thread(
            search_msn_instruments,
            q,
            market=normalized_market,
            asset_type=asset_type,
            limit=limit,
        )
        online = [item.to_dict() for item in remote]
    except Exception as exc:
        warnings.append("Microsoft Finance 在线搜索暂不可用：" + summarize_error(exc))

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*local, *online]:
        key = (item.get("market", normalized_market), item["symbol"].upper())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return {
        "query": q.strip(),
        "market": normalized_market,
        "asset_type": asset_type,
        "items": merged,
        "online": bool(online),
        "warnings": warnings,
    }


@app.get("/api/research/providers")
def research_providers() -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    for profile in LLM_PROVIDER_PROFILES.values():
        server_key = bool(
            (os.getenv(profile.api_key_env, "").strip() if profile.api_key_env else "")
            or _provider_environment_value(profile.provider_id, "API_KEY")
        )
        server_model = _provider_environment_value(profile.provider_id, "MODEL")
        payload = profile.public_dict(
            server_key_configured=server_key,
            server_model=server_model,
        )
        payload["base_url"] = (
            _provider_environment_value(profile.provider_id, "BASE_URL")
            or profile.base_url
        )
        providers.append(payload)
    return {"providers": providers, "max_concurrent_jobs": research_jobs.max_active}


@app.post("/api/research/providers/test")
async def test_research_provider(
    config: ProviderConnectionRequest,
) -> dict[str, Any]:
    client, runtime = _resolve_llm_runtime(config.options())
    if client is None:  # pragma: no cover - request always resolves online mode
        raise HTTPException(status_code=400, detail="连接测试需要在线模型。")
    started_at = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.complete,
            "你是 Raven Watch Agents Pro 的模型连接诊断器。",
            "仅回复 CONNECTED，不要添加其他内容。",
        )
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - started_at) * 1_000)
    runtime["usage"] = client.usage_summary()
    return {
        "status": "ok",
        "message": "模型连接成功",
        "latency_ms": latency_ms,
        "response_preview": response[:80],
        "llm": runtime,
    }


@app.get("/api/research/factors")
def research_factors() -> dict[str, Any]:
    references = factor_reference_catalog()
    return {
        "total_factors": len(FACTOR_WEIGHTS),
        "referenced_factors": len(references),
        "factors": references,
    }


@app.get("/api/dashboard")
async def dashboard(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
    period: str = Query("1y"),
    adjust: str = Query("qfq", pattern=r"^(none|qfq|hfq)$"),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    try:
        return await asyncio.to_thread(
            service.dashboard_payload,
            normalized_market,
            normalized_symbol,
            period,
            adjust,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.get("/api/quant/walk-forward")
async def quant_walk_forward(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
    period: str = Query("1y", pattern=r"^(3m|6m|1y|3y|5y)$"),
    adjust: str = Query("qfq", pattern=r"^(none|qfq|hfq)$"),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    try:
        return await asyncio.to_thread(
            service.quant_validation_payload,
            normalized_market,
            normalized_symbol,
            period,
            adjust,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.post("/api/quant/walk-forward")
async def configured_quant_walk_forward(
    request: QuantValidationRequest,
) -> dict[str, Any]:
    normalized_market = _normalize_market(request.market)
    normalized_symbol = _normalize_symbol(normalized_market, request.symbol)
    try:
        bars = await asyncio.to_thread(
            service.get_bars,
            normalized_market,
            normalized_symbol,
            request.period,
            request.adjust,
        )
        config = request.walk_forward_config(len(bars))
        return await asyncio.to_thread(
            service.quant_validation_payload,
            normalized_market,
            normalized_symbol,
            request.period,
            request.adjust,
            config,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.get("/api/evidence")
async def market_evidence(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
    period: str = Query("1y", pattern=r"^(3m|6m|1y|3y|5y)$"),
    adjust: str = Query("qfq", pattern=r"^(none|qfq|hfq)$"),
    limit: int = Query(12, ge=1, le=30),
    force: bool = Query(False),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    try:
        return await asyncio.to_thread(
            service.evidence_payload,
            normalized_market,
            normalized_symbol,
            period,
            adjust,
            limit=limit,
            force=force,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.post("/api/evidence")
async def configured_market_evidence(
    request: EvidenceRequest,
) -> dict[str, Any]:
    normalized_market = _normalize_market(request.market)
    normalized_symbol = _normalize_symbol(normalized_market, request.symbol)
    try:
        bars = await asyncio.to_thread(
            service.get_bars,
            normalized_market,
            normalized_symbol,
            request.period,
            request.adjust,
        )
        config = request.walk_forward_config(len(bars))
        return await asyncio.to_thread(
            service.evidence_payload,
            normalized_market,
            normalized_symbol,
            request.period,
            request.adjust,
            config=config,
            limit=request.limit,
            force=request.force,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.get("/api/market/live")
async def live_market(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    try:
        return await asyncio.to_thread(
            service.live_payload, normalized_market, normalized_symbol
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.get("/api/news")
async def market_news(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
    limit: int = Query(12, ge=1, le=30),
    force: bool = Query(False),
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    try:
        return await asyncio.to_thread(
            service.get_news,
            normalized_market,
            normalized_symbol,
            limit=limit,
            force=force,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.get("/api/stream", response_class=EventSourceResponse)
async def market_stream(
    market: str = Query("a-share"),
    symbol: str = Query("300750", min_length=1, max_length=32),
    refresh: int = Query(15, ge=5, le=120),
) -> AsyncIterable[ServerSentEvent]:
    normalized_market = _normalize_market(market)
    normalized_symbol = _normalize_symbol(normalized_market, symbol)
    while True:
        try:
            payload = await asyncio.to_thread(
                service.live_payload, normalized_market, normalized_symbol
            )
            yield ServerSentEvent(
                data=payload,
                event="market",
                retry=refresh * 1_000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ServerSentEvent(
                data={"error": summarize_error(exc)},
                event="market-error",
                retry=refresh * 1_000,
            )
        await asyncio.sleep(refresh)


@app.post("/api/research/agents")
async def run_agents(request: AgentResearchRequest) -> dict[str, Any]:
    market = _normalize_market(request.market)
    symbol = _normalize_symbol(market, request.symbol)
    options = request.options()
    _validate_agent_options(options)
    try:
        return await asyncio.to_thread(
            service.run_agents,
            market,
            symbol,
            options=options,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=summarize_error(exc)) from exc


@app.post("/api/research/jobs", status_code=202)
async def start_research_job(request: AgentResearchRequest) -> dict[str, Any]:
    market = _normalize_market(request.market)
    symbol = _normalize_symbol(market, request.symbol)
    options = request.options()
    _, _, runtime = _validate_agent_options(options)

    def runner(
        progress: Callable[[ProgressEvent], None],
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        return service.run_agents(
            market,
            symbol,
            options=options,
            progress=progress,
            cancel_event=cancel_event,
        )

    job = research_jobs.create(runner)
    total_steps = (
        len(options.selected_analysts)
        + options.debate_rounds * 2
        + 2
        + options.risk_rounds * 3
        + 1
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stream_url": f"/api/research/jobs/{job.job_id}/stream",
        "result_url": f"/api/research/jobs/{job.job_id}",
        "total_steps": total_steps,
        "mode": options.mode,
        "llm": runtime,
    }


@app.get("/api/research/jobs/{job_id}")
async def research_job(job_id: str) -> dict[str, Any]:
    job = research_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="智能体任务不存在或已过期。")
    return job.snapshot()


def _research_job_or_404(job_id: str) -> ResearchJob:
    job = research_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="智能体任务不存在或已过期。")
    return job


@app.get(
    "/api/research/jobs/{job_id}/stream",
    response_class=EventSourceResponse,
)
async def research_job_stream(
    request: Request,
    job: ResearchJob = Depends(_research_job_or_404),
) -> AsyncIterable[ServerSentEvent]:
    try:
        cursor = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        cursor = 0

    while True:
        batch, status = job.events_after(cursor)
        for item in batch:
            cursor = int(item["id"])
            yield ServerSentEvent(
                data=item["data"],
                event=str(item["event"]),
                id=str(item["id"]),
                retry=1_000,
            )
        if status in ResearchJobManager.TERMINAL_STATUSES:
            break
        if await request.is_disconnected():
            break
        await asyncio.sleep(0.2)


@app.delete("/api/research/jobs/{job_id}")
async def cancel_research_job(job_id: str) -> dict[str, Any]:
    job = research_jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="智能体任务不存在或已过期。")
    return job.snapshot()


if WEB_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = WEB_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=503, detail="Web 前端尚未构建。")
    return FileResponse(index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raven Watch Agents Pro Web Server")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        result = analyze_composite(generate_demo_ohlcv("ALPHA"))
        print(
            {
                "health": health(),
                "factor_count": len(result.factors),
                "strategy_count": len(result.strategies),
                "action": result.action,
            }
        )
        return
    import uvicorn

    uvicorn.run(
        "web_app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
