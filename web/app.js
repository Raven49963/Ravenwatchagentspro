(function () {
  "use strict";

  const terminal = document.getElementById("raven-terminal");
  const workspace = document.querySelector(".workspace");
  const symbolInput = document.getElementById("symbol-input");
  const symbolOptions = document.getElementById("symbol-options");
  const instrumentResults = document.getElementById("instrument-results");
  const instrumentCatalogDialog = document.getElementById("instrument-catalog-dialog");
  const catalogQueryInput = document.getElementById("catalog-query-input");
  const catalogAssetSelect = document.getElementById("catalog-asset-select");
  const catalogCategorySelect = document.getElementById("catalog-category-select");
  const assetTypeSelect = document.getElementById("asset-type-select");
  const periodSelect = document.getElementById("period-select");
  const refreshButton = document.getElementById("refresh-button");
  const refreshNewsButton = document.getElementById("refresh-news-button");
  const runAgentsButton = document.getElementById("run-agents-button");
  const quantSettingsDialog = document.getElementById("quant-settings-dialog");
  const quantSettingsForm = document.getElementById("quant-settings-form");
  const agentSettingsDialog = document.getElementById("agent-settings-dialog");
  const agentSettingsForm = document.getElementById("agent-settings-form");
  const reportsDialog = document.getElementById("reports-dialog");
  const reportTabs = document.getElementById("report-tabs");
  const reportContent = document.getElementById("report-content");
  const toast = document.getElementById("toast");
  const initialQuery = new URLSearchParams(window.location.search);
  const desktopToken = initialQuery.get("desktop_token") || "";
  const snapshotMode = initialQuery.get("snapshot") === "1";
  if (desktopToken) {
    initialQuery.delete("desktop_token");
    const cleanQuery = initialQuery.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${cleanQuery ? `?${cleanQuery}` : ""}`);
  }

  const state = {
    market: "a-share",
    symbol: "300750",
    period: "1y",
    assetType: "all",
    data: null,
    stream: null,
    requestId: 0,
    newsRequestId: 0,
    newsKey: "",
    newsData: null,
    evidenceData: null,
    quantConfig: {
      train_rows: 0,
      test_rows: 0,
      commission_bps: 3,
      slippage_bps: 2,
      stress_multiplier: 2,
      max_position_percent: 100,
      bootstrap_horizon: 63,
      bootstrap_simulations: 1000,
      bootstrap_block_size: 5,
      random_seed: 7,
    },
    charts: {},
    toastTimer: null,
    desktopToken,
    providers: [],
    agentMode: "offline",
    researchJob: null,
    researchStream: null,
    agentPlan: [],
    researchResult: null,
    searchTimer: null,
    searchController: null,
    searchIndex: -1,
    catalogMarket: "a-share",
    catalogAssetType: "all",
    catalogCategory: "all",
    catalogPage: 1,
    catalogPages: 1,
    catalogTimer: null,
    catalogController: null,
    catalogPayload: null,
  };

  const marketDefaults = {
    "a-share": "300750",
    nasdaq: "NVDA",
    hk: "0700.HK",
    global: "7203.T",
  };

  const marketLabels = {
    "a-share": "A 股",
    nasdaq: "美股",
    hk: "港股",
    global: "全球市场",
  };

  const currencySymbols = {
    CNY: "¥",
    USD: "$",
    HKD: "HK$",
    JPY: "¥",
    EUR: "€",
    GBP: "£",
    GBp: "p",
    CAD: "C$",
    AUD: "A$",
    CHF: "CHF ",
    INR: "₹",
    SGD: "S$",
    KRW: "₩",
    TWD: "NT$",
  };

  const factorLabels = {
    trend: "趋势",
    momentum: "动量/突破",
    reversal: "反转",
    risk: "风险",
    liquidity: "流动性",
    flow: "资金流",
    session: "时段结构",
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function safeNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function safeExternalUrl(value) {
    try {
      const target = new URL(String(value || ""));
      return ["http:", "https:"].includes(target.protocol) ? target.href : "";
    } catch (_error) {
      return "";
    }
  }

  function createIcon(name) {
    const icon = document.createElement("i");
    icon.dataset.lucide = name;
    return icon;
  }

  function refreshIcons() {
    if (window.lucide) window.lucide.createIcons();
  }

  function createSourceLink(url, label, className = "source-link") {
    const safeUrl = safeExternalUrl(url);
    if (!safeUrl) return null;
    const link = document.createElement("a");
    link.className = className;
    link.href = safeUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = label || "查看研究来源";
    link.setAttribute("aria-label", label || "查看研究来源");
    link.append(createIcon("book-open"));
    return link;
  }

  function priceDigits(price) {
    return Math.abs(price) >= 100 ? 2 : 3;
  }

  function formatPrice(value) {
    const number = safeNumber(value, NaN);
    return Number.isFinite(number) ? number.toFixed(priceDigits(number)) : "--";
  }

  function formatPercent(value, digits = 2) {
    const number = safeNumber(value, NaN);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "--";
  }

  function formatCompact(value, currency = false, currencyCode = "") {
    const number = safeNumber(value, NaN);
    if (!Number.isFinite(number)) return "--";
    const absolute = Math.abs(number);
    const units = ["a-share", "hk"].includes(state.market)
      ? [[1e8, "亿"], [1e4, "万"]]
      : [[1e9, "B"], [1e6, "M"], [1e3, "K"]];
    const prefix = currency
      ? (currencySymbols[currencyCode] || (currencyCode ? `${currencyCode} ` : ""))
      : "";
    for (const [scale, suffix] of units) {
      if (absolute >= scale) {
        return `${prefix}${(number / scale).toFixed(2)}${suffix}`;
      }
    }
    return `${prefix}${number.toFixed(0)}`;
  }

  function formatTime(value) {
    if (!value) return "--";
    const timestamp = new Date(value);
    if (Number.isNaN(timestamp.getTime())) return String(value);
    return timestamp.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function actionLabel(action) {
    return { BUY: "买入", HOLD: "观望", SELL: "减仓" }[action] || action;
  }

  function authenticatedUrl(url) {
    if (!state.desktopToken) return url;
    const target = new URL(url, window.location.origin);
    target.searchParams.set("desktop_token", state.desktopToken);
    return `${target.pathname}${target.search}${target.hash}`;
  }

  function apiErrorMessage(payload, status) {
    if (!payload || payload.detail == null) return `请求失败 (${status})`;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((item) => (item && item.msg ? item.msg : String(item)))
        .join("；");
    }
    return String(payload.detail);
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(authenticatedUrl(url), {
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(apiErrorMessage(payload, response.status));
    }
    return payload;
  }

  function setStatus(message, online, detail) {
    const status = byId("status-message");
    status.innerHTML = "";
    const dot = document.createElement("b");
    status.append(dot, document.createTextNode(` ${message}`));
    status.classList.toggle("is-online", Boolean(online));
    byId("status-detail").textContent = detail || "--";
    byId("sidebar-connection").textContent = online ? "实时" : "重连中";
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 3800);
  }

  function setBusy(busy) {
    workspace.setAttribute("aria-busy", String(busy));
    refreshButton.disabled = busy;
    const icon = refreshButton.querySelector("svg");
    if (icon) icon.classList.toggle("spin", busy);
  }

  function initializeCharts() {
    if (!window.echarts) {
      byId("chart-empty").classList.remove("is-hidden");
      showToast("图表引擎加载失败，请检查网络连接。");
      return;
    }
    state.charts.kline = window.echarts.init(byId("kline-chart"), null, { renderer: "canvas" });
    state.charts.intraday = window.echarts.init(byId("intraday-chart"), null, { renderer: "canvas" });
    state.charts.heatmap = window.echarts.init(byId("factor-heatmap"), null, { renderer: "canvas" });
    state.charts.validation = window.echarts.init(byId("validation-chart"), null, { renderer: "canvas" });

    const observer = new ResizeObserver(() => {
      Object.values(state.charts).forEach((chart) => chart && chart.resize());
    });
    observer.observe(document.querySelector(".terminal-shell"));
  }

  function chartColors() {
    const isChina = state.market === "a-share";
    return {
      up: isChina ? "#ff6672" : "#24d6a0",
      down: isChina ? "#24d6a0" : "#ff6672",
      text: "#8ea29e",
      line: "#344548",
      accent: "#24d6a0",
      warning: "#f2c14e",
      info: "#55a7ff",
      violet: "#9a8cff",
    };
  }

  function calculateMovingAverage(candles, windowSize) {
    return candles.map((_item, index) => {
      if (index < windowSize - 1) return "-";
      let total = 0;
      for (let cursor = 0; cursor < windowSize; cursor += 1) {
        total += safeNumber(candles[index - cursor].close);
      }
      return +(total / windowSize).toFixed(4);
    });
  }

  function renderKline() {
    const chart = state.charts.kline;
    const candles = state.data && state.data.candles ? state.data.candles : [];
    if (!chart || !candles.length) return;
    const colors = chartColors();
    const dates = candles.map((item) => item.date);
    const candleData = candles.map((item) => [item.open, item.close, item.low, item.high]);
    const volumeData = candles.map((item, index) => ({
      value: item.volume,
      itemStyle: { color: item.close >= item.open ? colors.up : colors.down, opacity: 0.66 },
      name: dates[index],
    }));
    const start = candles.length > 130 ? Math.max(0, 100 - (130 / candles.length) * 100) : 0;

    chart.setOption({
      animation: false,
      backgroundColor: "transparent",
      axisPointer: { link: [{ xAxisIndex: "all" }], label: { backgroundColor: "#263436" } },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        borderColor: colors.line,
        backgroundColor: "rgba(16,23,25,0.96)",
        textStyle: { color: "#e8f1ef", fontSize: 11 },
      },
      legend: {
        top: 9,
        left: 52,
        itemWidth: 14,
        itemHeight: 2,
        textStyle: { color: colors.text, fontSize: 9 },
        data: ["MA5", "MA20", "MA60"],
      },
      grid: [
        { left: 54, right: 62, top: 36, height: 306 },
        { left: 54, right: 62, top: 366, height: 66 },
      ],
      xAxis: [
        {
          type: "category",
          data: dates,
          boundaryGap: true,
          axisLine: { lineStyle: { color: colors.line } },
          axisLabel: { color: colors.text, fontSize: 9, hideOverlap: true },
          axisTick: { show: false },
          splitLine: { show: false },
          min: "dataMin",
          max: "dataMax",
        },
        {
          type: "category",
          gridIndex: 1,
          data: dates,
          boundaryGap: true,
          axisLine: { lineStyle: { color: colors.line } },
          axisLabel: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          min: "dataMin",
          max: "dataMax",
        },
      ],
      yAxis: [
        {
          scale: true,
          position: "right",
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: colors.text, fontSize: 9 },
          splitLine: { lineStyle: { color: colors.line, opacity: 0.55 } },
        },
        {
          scale: true,
          gridIndex: 1,
          position: "right",
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: colors.text, fontSize: 8, formatter: (value) => formatCompact(value) },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1], start, end: 100 },
        {
          type: "slider",
          xAxisIndex: [0, 1],
          top: 449,
          height: 18,
          start,
          end: 100,
          borderColor: colors.line,
          backgroundColor: "#101719",
          fillerColor: "rgba(36,214,160,0.12)",
          handleStyle: { color: colors.accent, borderColor: colors.accent },
          textStyle: { color: colors.text, fontSize: 8 },
          dataBackground: { lineStyle: { color: colors.text, opacity: 0.35 }, areaStyle: { opacity: 0 } },
          selectedDataBackground: { lineStyle: { color: colors.accent, opacity: 0.7 }, areaStyle: { opacity: 0 } },
        },
      ],
      series: [
        {
          name: "K线",
          type: "candlestick",
          data: candleData,
          itemStyle: {
            color: colors.up,
            color0: colors.down,
            borderColor: colors.up,
            borderColor0: colors.down,
          },
        },
        { name: "MA5", type: "line", data: calculateMovingAverage(candles, 5), smooth: true, symbol: "none", lineStyle: { width: 1.2, color: colors.warning } },
        { name: "MA20", type: "line", data: calculateMovingAverage(candles, 20), smooth: true, symbol: "none", lineStyle: { width: 1.2, color: colors.info } },
        { name: "MA60", type: "line", data: calculateMovingAverage(candles, 60), smooth: true, symbol: "none", lineStyle: { width: 1.1, color: colors.violet } },
        { name: "成交量", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: volumeData, barMaxWidth: 8 },
      ],
    }, true);
  }

  function renderIntraday() {
    const chart = state.charts.intraday;
    const rows = state.data && state.data.intraday ? state.data.intraday : [];
    if (!chart || !rows.length) {
      if (chart) chart.clear();
      return;
    }
    const colors = chartColors();
    const times = rows.map((item) => item.time);
    chart.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        borderColor: colors.line,
        backgroundColor: "rgba(16,23,25,0.96)",
        textStyle: { color: "#e8f1ef", fontSize: 11 },
      },
      legend: {
        top: 10,
        left: 52,
        itemWidth: 14,
        itemHeight: 2,
        textStyle: { color: colors.text, fontSize: 9 },
        data: ["价格", "均价"],
      },
      grid: [
        { left: 54, right: 62, top: 40, height: 300 },
        { left: 54, right: 62, top: 365, height: 72 },
      ],
      xAxis: [
        {
          type: "category",
          data: times,
          boundaryGap: false,
          axisLine: { lineStyle: { color: colors.line } },
          axisTick: { show: false },
          axisLabel: {
            color: colors.text,
            fontSize: 9,
            formatter: (value) => value.slice(11, 16),
            hideOverlap: true,
          },
        },
        {
          type: "category",
          gridIndex: 1,
          data: times,
          boundaryGap: true,
          axisLabel: { show: false },
          axisLine: { lineStyle: { color: colors.line } },
          axisTick: { show: false },
        },
      ],
      yAxis: [
        {
          scale: true,
          position: "right",
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: colors.text, fontSize: 9 },
          splitLine: { lineStyle: { color: colors.line, opacity: 0.55 } },
        },
        {
          gridIndex: 1,
          position: "right",
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: colors.text, fontSize: 8, formatter: (value) => formatCompact(value) },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "价格",
          type: "line",
          data: rows.map((item) => item.price),
          showSymbol: false,
          lineStyle: { width: 1.8, color: colors.accent },
          areaStyle: { color: "rgba(36,214,160,0.08)" },
        },
        {
          name: "均价",
          type: "line",
          data: rows.map((item) => item.average_price),
          showSymbol: false,
          lineStyle: { width: 1.1, color: colors.warning },
        },
        {
          name: "分钟量",
          type: "bar",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: rows.map((item) => item.volume),
          itemStyle: { color: colors.info, opacity: 0.45 },
          barMaxWidth: 4,
        },
      ],
    }, true);
  }

  function freshnessLabel(snapshot) {
    const status = snapshot.session_status || "closed";
    const delay = Number(snapshot.delayed_seconds);
    if (status === "open") return delay > 60 ? `盘中 · 约 ${Math.ceil(delay / 60)} 分钟` : "盘中 · 近实时";
    if (status === "delayed") return Number.isFinite(delay) ? `行情延迟约 ${Math.max(1, Math.ceil(delay / 60))} 分钟` : "延迟行情";
    return "休市 · 最近成交";
  }

  function renderSnapshot(snapshot, provider) {
    if (!snapshot) return;
    const price = safeNumber(snapshot.price);
    const change = safeNumber(snapshot.change);
    const changePct = safeNumber(snapshot.change_pct);
    byId("quote-name").textContent = snapshot.name || state.symbol;
    byId("quote-symbol").textContent = snapshot.symbol || state.symbol;
    byId("quote-market-meta").textContent = [
      snapshot.exchange || marketLabels[state.market],
      snapshot.asset_type_label || "证券",
      snapshot.currency,
    ].filter(Boolean).join(" · ");
    byId("quote-price").textContent = formatPrice(price);
    const changeElement = byId("quote-change");
    changeElement.textContent = `${change >= 0 ? "+" : ""}${formatPrice(change)}  ${changePct >= 0 ? "+" : ""}${formatPercent(changePct)}`;
    changeElement.classList.remove("is-up", "is-down");
    changeElement.classList.add(change >= 0 ? "is-up" : "is-down");

    byId("tape-open").textContent = formatPrice(snapshot.open);
    byId("tape-high").textContent = formatPrice(snapshot.high);
    byId("tape-low").textContent = formatPrice(snapshot.low);
    byId("tape-volume").textContent = formatCompact(snapshot.volume);
    byId("tape-amount").textContent = formatCompact(snapshot.amount, true, snapshot.currency || "");
    const providerLink = byId("tape-provider");
    const providerLabel = (provider && provider.realtime_label) || snapshot.provider || "--";
    const providerUrl = safeExternalUrl((provider && provider.realtime_url) || snapshot.source_url);
    providerLink.textContent = providerLabel;
    providerLink.title = providerUrl ? `打开 ${providerLabel}` : providerLabel;
    if (providerUrl) providerLink.href = providerUrl;
    else providerLink.removeAttribute("href");
    byId("tape-time").textContent = formatTime(snapshot.timestamp);
    const freshness = byId("quote-freshness");
    freshness.textContent = freshnessLabel(snapshot);
    freshness.className = `data-freshness is-${snapshot.session_status || "closed"}`;
    byId("sidebar-provider").textContent = `${providerLabel} · 15 秒自动刷新`;
  }

  function renderDataQuality(quality) {
    const badge = byId("data-quality-badge");
    if (!quality) {
      badge.textContent = "样本未知";
      badge.title = "当前未返回数据质量摘要";
      badge.classList.add("is-limited");
      return;
    }
    const rows = Math.max(0, Number(quality.rows) || 0);
    const completeness = Math.max(0, Math.min(1, Number(quality.completeness) || 0));
    badge.textContent = `样本 ${rows.toLocaleString("zh-CN")} · ${quality.status || "未知"}`;
    badge.title = [
      quality.start && quality.end ? `覆盖 ${quality.start} 至 ${quality.end}` : "",
      `完整度 ${(completeness * 100).toFixed(1)}%`,
      `质量分 ${Number(quality.score) || 0}/100`,
      quality.adjustment ? `复权 ${quality.adjustment}` : "",
    ].filter(Boolean).join("；");
    badge.classList.toggle(
      "is-limited",
      rows < 120 || completeness < 0.99 || Number(quality.invalid_rows) > 0,
    );
  }

  function renderDecision(analysis) {
    if (!analysis) return;
    const score = safeNumber(analysis.score);
    const progress = Math.min(1, Math.abs(score) / 100);
    const circumference = 314;
    const ring = byId("score-ring-value");
    ring.style.strokeDashoffset = String(circumference * (1 - progress));
    ring.style.stroke = score > 15 ? "var(--accent)" : score < -15 ? "var(--negative)" : "var(--warning)";
    byId("decision-score").textContent = `${score > 0 ? "+" : ""}${score.toFixed(1)}`;
    byId("decision-label").textContent = analysis.action_label;
    byId("decision-label").style.color = score > 15 ? "var(--accent)" : score < -15 ? "var(--negative)" : "var(--warning)";
    byId("decision-summary").textContent = analysis.summary;
    byId("decision-confidence").textContent = `${analysis.confidence}%`;
    byId("decision-position").textContent = formatPercent(analysis.target_position, 0);
    byId("decision-risk").textContent = analysis.risk_level;
    byId("decision-stop").textContent = formatPercent(analysis.stop_loss_pct);
    byId("decision-take").textContent = formatPercent(analysis.take_profit_pct);
    byId("regime-name").textContent = analysis.regime.name;
    byId("regime-description").textContent = analysis.regime.description;
    byId("regime-badge").textContent = analysis.regime.name;
    byId("strategy-regime").textContent = `${analysis.regime.name} · 动态权重`;
  }

  function renderFactors(factors) {
    const container = byId("factor-list");
    container.replaceChildren();
    const availableCount = factors.filter((factor) => factor.available !== false).length;
    byId("factor-count").textContent = `${availableCount} / ${factors.length} 可用`;
    factors.forEach((factor) => {
      const available = factor.available !== false;
      const row = document.createElement("div");
      row.className = `factor-row${available ? "" : " is-unavailable"}`;
      row.title = [
        factor.description,
        factor.formula ? `公式：${factor.formula}` : "",
        factor.reference_title ? `来源：${factor.reference_title}` : "",
      ].filter(Boolean).join("\n");

      const name = document.createElement("div");
      name.className = "factor-name";
      const label = document.createElement("span");
      label.className = "factor-label";
      const labelText = document.createElement("b");
      labelText.textContent = factor.name;
      label.append(labelText);
      const sourceLink = createSourceLink(
        factor.reference_url,
        factor.reference_title,
        "factor-source-link",
      );
      if (sourceLink) label.append(sourceLink);
      const category = document.createElement("small");
      category.textContent = available
        ? factor.category
        : `${factor.category} · 需 ${safeNumber(factor.history_required, 1)} 根`;
      name.append(label, category);

      const track = document.createElement("div");
      track.className = `factor-track${available ? "" : " is-unavailable"}`;
      const score = safeNumber(factor.score);
      if (available) {
        const bar = document.createElement("span");
        bar.className = `factor-bar${score < 0 ? " is-negative" : ""}`;
        const width = Math.min(50, Math.abs(score) / 2);
        bar.style.width = `${width}%`;
        bar.style.left = score >= 0 ? "50%" : `${50 - width}%`;
        track.append(bar);
      }

      const value = document.createElement("strong");
      value.className = `factor-score ${score > 15 ? "is-positive" : score < -15 ? "is-negative" : ""}`;
      value.textContent = available ? `${score > 0 ? "+" : ""}${score.toFixed(0)}` : "N/A";
      row.append(name, track, value);
      container.append(row);
    });
    refreshIcons();
  }

  function renderFactorHeatmap(history) {
    const chart = state.charts.heatmap;
    if (!chart || !history.length) return;
    const keys = Object.keys(factorLabels);
    const dates = history.map((item) => item.date.slice(5));
    const data = [];
    history.forEach((item, x) => {
      keys.forEach((key, y) => data.push([x, y, safeNumber(item[key])]));
    });
    chart.setOption({
      animation: false,
      tooltip: {
        position: "top",
        borderColor: "#344548",
        backgroundColor: "rgba(16,23,25,0.96)",
        textStyle: { color: "#e8f1ef", fontSize: 10 },
        formatter: (params) => `${history[params.value[0]].date}<br>${factorLabels[keys[params.value[1]]]}: ${params.value[2].toFixed(1)}`,
      },
      grid: { top: 24, left: 66, right: 18, bottom: 62 },
      xAxis: {
        type: "category",
        data: dates,
        splitArea: { show: false },
        axisLine: { lineStyle: { color: "#344548" } },
        axisTick: { show: false },
        axisLabel: { color: "#8ea29e", fontSize: 8, interval: 4, rotate: 45 },
      },
      yAxis: {
        type: "category",
        data: keys.map((key) => factorLabels[key]),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#8ea29e", fontSize: 9 },
      },
      visualMap: {
        min: -100,
        max: 100,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 8,
        itemWidth: 90,
        itemHeight: 5,
        text: ["多", "空"],
        textStyle: { color: "#8ea29e", fontSize: 8 },
        inRange: { color: ["#ff6672", "#263436", "#24d6a0"] },
      },
      series: [{
        name: "因子得分",
        type: "heatmap",
        data,
        itemStyle: { borderWidth: 2, borderColor: "#101719", borderRadius: 1 },
        emphasis: { itemStyle: { borderColor: "#e8f1ef", borderWidth: 1 } },
      }],
    }, true);
  }

  function renderStrategies(strategies) {
    const container = byId("strategy-list");
    container.replaceChildren();
    strategies.forEach((strategy) => {
      const score = safeNumber(strategy.score);
      const row = document.createElement("div");
      row.className = "strategy-row";
      row.title = strategy.rationale;

      const name = document.createElement("div");
      name.className = "strategy-name";
      const strong = document.createElement("strong");
      strong.textContent = strategy.name;
      const weight = document.createElement("span");
      weight.textContent = `权重 ${(safeNumber(strategy.weight) * 100).toFixed(0)}%`;
      name.append(strong, weight);

      const track = document.createElement("div");
      track.className = "strategy-track";
      const bar = document.createElement("div");
      bar.className = `strategy-bar${score < 0 ? " is-negative" : ""}`;
      bar.style.width = `${Math.min(100, Math.abs(score))}%`;
      track.append(bar);

      const vote = document.createElement("div");
      vote.className = "strategy-vote";
      const action = document.createElement("strong");
      action.className = score > 20 ? "is-positive" : score < -20 ? "is-negative" : "";
      action.textContent = actionLabel(strategy.action);
      const value = document.createElement("span");
      value.textContent = `${score > 0 ? "+" : ""}${score.toFixed(1)}`;
      vote.append(action, value);
      row.append(name, track, vote);
      container.append(row);
    });
  }

  function setValidationMetric(id, value, tone = "") {
    const element = byId(id);
    element.textContent = value;
    element.classList.remove("is-positive", "is-negative");
    if (tone) element.classList.add(tone);
  }

  function renderValidation(validation) {
    const chart = state.charts.validation;
    const body = byId("validation-fold-body");
    body.replaceChildren();
    if (!validation || validation.available === false) {
      byId("validation-profile").textContent = "历史数据不足";
      byId("validation-badge").textContent = "暂不可用";
      byId("validation-summary").textContent = validation?.message || "滚动样本外结果暂不可用。";
      [
        "validation-return",
        "validation-excess",
        "validation-drawdown",
        "validation-psr",
        "validation-loss-risk",
        "validation-position",
      ].forEach((id) => setValidationMetric(id, "--"));
      if (chart) chart.clear();
      return;
    }

    const metrics = validation.metrics || {};
    const benchmark = validation.benchmark_metrics || {};
    const bootstrap = validation.bootstrap || {};
    const totalReturn = safeNumber(metrics.total_return);
    const excess = totalReturn - safeNumber(benchmark.total_return);
    const psr = safeNumber(validation.probabilistic_sharpe);
    const lossRisk = safeNumber(bootstrap.loss_probability);
    const position = safeNumber(validation.latest_position);
    setValidationMetric("validation-return", formatPercent(totalReturn), totalReturn >= 0 ? "is-positive" : "is-negative");
    setValidationMetric("validation-excess", formatPercent(excess), excess >= 0 ? "is-positive" : "is-negative");
    setValidationMetric("validation-drawdown", formatPercent(metrics.max_drawdown), "is-negative");
    setValidationMetric("validation-psr", formatPercent(psr, 0), psr >= 0.75 ? "is-positive" : psr < 0.5 ? "is-negative" : "");
    setValidationMetric("validation-loss-risk", formatPercent(lossRisk, 0), lossRisk <= 0.35 ? "is-positive" : lossRisk > 0.55 ? "is-negative" : "");
    setValidationMetric("validation-position", formatPercent(position, 0));
    byId("validation-profile").textContent = `${validation.latest_profile} · ${validation.folds.length} 折`;
    byId("validation-badge").textContent = `${validation.verdict} · ${safeNumber(validation.robustness_score, 0).toFixed(0)}`;
    byId("validation-summary").textContent = validation.summary;
    const config = validation.config || {};
    document.querySelector(".validation-panel").title = [
      `训练 ${safeNumber(config.train_rows, 0)} 日 / 测试 ${safeNumber(config.test_rows, 0)} 日`,
      "次日执行",
      `单边成本 ${formatPercent(config.one_way_cost)}`,
      `压力倍数 ${safeNumber(config.stress_multiplier, 1).toFixed(1)}x`,
      `Bootstrap ${safeNumber(bootstrap.simulations, 0).toLocaleString("zh-CN")} 次`,
    ].join(" · ");
    byId("quant-settings-status").textContent = [
      `${safeNumber(config.train_rows, 0)} / ${safeNumber(config.test_rows, 0)} 日`,
      `成本 ${formatPercent(config.one_way_cost)}`,
      `仓位 ${formatPercent(config.max_position, 0)}`,
      `${safeNumber(bootstrap.simulations, 0).toLocaleString("zh-CN")} 次重采样`,
    ].join(" · ");

    const fragment = document.createDocumentFragment();
    validation.folds.forEach((fold) => {
      const row = document.createElement("tr");
      const period = document.createElement("td");
      period.textContent = `${fold.test_start.slice(2)} → ${fold.test_end.slice(2)}`;
      period.title = `${fold.test_start} 至 ${fold.test_end}`;
      const profile = document.createElement("td");
      profile.textContent = fold.selected_name;
      const strategy = document.createElement("td");
      strategy.textContent = formatPercent(fold.test_return);
      strategy.className = safeNumber(fold.test_return) >= 0 ? "is-positive" : "is-negative";
      const baseline = document.createElement("td");
      baseline.textContent = formatPercent(fold.benchmark_return);
      const relative = document.createElement("td");
      relative.textContent = formatPercent(fold.excess_return);
      relative.className = safeNumber(fold.excess_return) >= 0 ? "is-positive" : "is-negative";
      row.append(period, profile, strategy, baseline, relative);
      fragment.append(row);
    });
    body.append(fragment);

    const curve = validation.equity_curve || [];
    if (!chart || !curve.length) return;
    const colors = chartColors();
    chart.setOption({
      animation: false,
      color: [colors.accent, colors.info, colors.warning],
      tooltip: {
        trigger: "axis",
        backgroundColor: "#172124",
        borderColor: colors.line,
        textStyle: { color: "#d9e5e2", fontSize: 10 },
        valueFormatter: (value) => safeNumber(value).toFixed(2),
      },
      legend: {
        top: 10,
        right: 14,
        textStyle: { color: colors.text, fontSize: 9 },
        itemWidth: 14,
        itemHeight: 2,
      },
      grid: { left: 48, right: 18, top: 44, bottom: 34 },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: curve.map((item) => item.date),
        axisLine: { lineStyle: { color: colors.line } },
        axisLabel: { color: colors.text, fontSize: 9, hideOverlap: true },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: colors.text, fontSize: 9, formatter: (value) => safeNumber(value).toFixed(0) },
        splitLine: { lineStyle: { color: colors.line, opacity: 0.45 } },
      },
      series: [
        {
          name: "样本外策略",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2 },
          data: curve.map((item) => item.strategy),
        },
        {
          name: "买入持有",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 1.4 },
          data: curve.map((item) => item.benchmark),
        },
        {
          name: "双倍成本",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 1.2, type: "dashed" },
          data: curve.map((item) => item.stress),
        },
      ],
    }, true);
  }

  function renderMining(results) {
    const body = byId("mining-table-body");
    body.replaceChildren();
    byId("mining-count").textContent = `${results.length} 序列 · 5日前瞻`;
    results.forEach((item) => {
      const row = document.createElement("tr");
      const nameCell = document.createElement("td");
      const name = document.createElement("span");
      name.textContent = item.name;
      nameCell.append(name);
      const sourceLink = createSourceLink(
        item.reference_url,
        item.reference_title,
        "table-source-link",
      );
      if (sourceLink) nameCell.append(sourceLink);
      row.append(nameCell);

      const values = [
        safeNumber(item.information_coefficient).toFixed(3),
        safeNumber(item.t_statistic).toFixed(2),
        formatPercent(item.directional_win_rate, 1),
        String(item.observations),
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index === 0) {
          const ic = safeNumber(item.information_coefficient);
          cell.className = ic > 0.03 ? "is-positive" : ic < -0.03 ? "is-negative" : "";
        }
        row.append(cell);
      });
      body.append(row);
    });
    refreshIcons();
  }

  function renderFundamentalProviders(snapshot) {
    const container = byId("fundamental-source-status");
    container.replaceChildren();
    const providers = snapshot?.providers || [];
    providers.forEach((provider) => {
      const status = ["ok", "empty", "error", "timeout"].includes(provider.status)
        ? provider.status
        : "error";
      const sourceUrl = safeExternalUrl(provider.source_url);
      const element = document.createElement(sourceUrl ? "a" : "span");
      element.className = `fundamental-provider is-${status}`;
      element.title = provider.message || `${provider.credibility || "公开数据源"} · ${provider.source_kind || "基本面"}`;
      if (sourceUrl) {
        element.href = sourceUrl;
        element.target = "_blank";
        element.rel = "noopener noreferrer";
      }
      const count = status === "ok" ? `${safeNumber(provider.field_count)} 项` : status === "empty" ? "无结果" : status === "timeout" ? "超时" : "失败";
      element.append(document.createElement("b"), document.createTextNode(`${provider.label} · ${count}`));
      container.append(element);
    });
    if (!container.childElementCount) {
      const unavailable = document.createElement("span");
      unavailable.className = "fundamental-provider is-error";
      unavailable.append(document.createElement("b"), document.createTextNode("基本面数据源不可用"));
      container.append(unavailable);
    }
    const warnings = snapshot?.warnings || [];
    if (warnings.length) container.title = warnings.join("；");
    else container.removeAttribute("title");
  }

  function renderFundamentals(snapshot, assessment) {
    const metrics = assessment?.metrics || [];
    const reportDate = assessment?.report_date || "报告期未知";
    byId("fundamental-meta").textContent = assessment?.available
      ? `${assessment.company || state.symbol} · ${safeNumber(assessment.available_metrics)} / ${safeNumber(assessment.total_metrics)} · ${reportDate}`
      : "数据覆盖不足";
    const container = byId("fundamental-metrics");
    container.replaceChildren();
    metrics.forEach((metric) => {
      const item = document.createElement("div");
      item.className = `fundamental-metric${metric.available ? "" : " is-unavailable"}`;
      const label = document.createElement("span");
      label.textContent = metric.label;
      const value = document.createElement("strong");
      value.textContent = metric.display || "--";
      if (metric.tone === "positive") value.className = "is-positive";
      if (metric.tone === "negative") value.className = "is-negative";
      const source = document.createElement("small");
      source.textContent = metric.source || "暂无可用字段";
      item.append(label, value, source);
      container.append(item);
    });
    if (!metrics.length) {
      const item = document.createElement("div");
      item.className = "fundamental-metric is-unavailable";
      item.append(document.createElement("span"), document.createElement("strong"), document.createElement("small"));
      item.children[0].textContent = "基本面";
      item.children[1].textContent = "--";
      item.children[2].textContent = "数据暂不可用";
      container.append(item);
    }
    renderFundamentalProviders(snapshot || {});
  }

  function renderEvidence(payload) {
    state.evidenceData = payload;
    const local = payload.local_evidence || {};
    const fundamentals = payload.fundamental_assessment || {};
    const news = payload.news_assessment || {};
    const score = safeNumber(local.score, NaN);
    const scoreElement = byId("evidence-score");
    scoreElement.textContent = Number.isFinite(score) ? `${score > 0 ? "+" : ""}${score.toFixed(1)}` : "--";
    scoreElement.classList.remove("is-positive", "is-negative");
    if (score >= 8) scoreElement.classList.add("is-positive");
    if (score <= -8) scoreElement.classList.add("is-negative");
    byId("evidence-label").textContent = local.label || "证据不足";
    byId("evidence-badge").textContent = `${local.label || "不可用"} · ${safeNumber(local.confidence).toFixed(0)}`;
    byId("evidence-confidence").textContent = `${safeNumber(local.confidence).toFixed(0)}%`;
    byId("evidence-coverage").textContent = formatPercent(local.coverage, 0);
    byId("evidence-summary").textContent = local.summary || "本地证据暂不可用。";
    const fundamentalProviders = payload.fundamentals?.providers || [];
    const providerCount = fundamentalProviders.filter((provider) => provider.status === "ok").length;
    byId("evidence-meta").textContent = `${providerCount} 个基本面源 · ${safeNumber(news.article_count)} 条事件`;

    const componentList = byId("evidence-component-list");
    componentList.replaceChildren();
    (local.components || []).forEach((component) => {
      const row = document.createElement("div");
      row.className = `evidence-component-row${component.available ? "" : " is-unavailable"}`;
      const name = document.createElement("div");
      name.className = "evidence-component-name";
      const title = document.createElement("strong");
      title.textContent = component.label;
      const detail = document.createElement("span");
      detail.textContent = component.available
        ? `可信 ${safeNumber(component.confidence).toFixed(0)}% · 权重 ${formatPercent(component.effective_weight, 0)}`
        : "未纳入";
      name.append(title, detail);
      const track = document.createElement("div");
      track.className = "evidence-component-track";
      const bar = document.createElement("b");
      const componentScore = safeNumber(component.score);
      bar.style.width = `${Math.min(100, Math.abs(componentScore))}%`;
      if (componentScore < 0) bar.className = "is-negative";
      track.append(bar);
      const value = document.createElement("strong");
      value.className = "evidence-component-value";
      value.textContent = component.available ? `${componentScore > 0 ? "+" : ""}${componentScore.toFixed(1)}` : "--";
      if (componentScore >= 20) value.classList.add("is-positive");
      if (componentScore <= -20) value.classList.add("is-negative");
      row.title = component.detail || component.label;
      row.append(name, track, value);
      componentList.append(row);
    });

    byId("news-positive-count").textContent = String(safeNumber(news.positive_count));
    byId("news-negative-count").textContent = String(safeNumber(news.negative_count));
    byId("news-neutral-count").textContent = String(safeNumber(news.neutral_count));
    const catalyst = (news.catalysts || [])[0];
    const risk = (news.risks || [])[0];
    byId("news-top-catalyst").textContent = catalyst?.title || "暂无显著正面事件";
    byId("news-top-catalyst").title = catalyst?.title || "";
    byId("news-top-risk").textContent = risk?.title || "暂无显著负面事件";
    byId("news-top-risk").title = risk?.title || "";
    renderFundamentals(payload.fundamentals || {}, fundamentals);
  }

  function setEvidenceLoading(loading, reset = false) {
    if (!reset) return;
    byId("evidence-meta").textContent = loading ? "正在获取" : "等待证据";
    byId("evidence-badge").textContent = loading ? "计算中" : "暂不可用";
    if (!loading) return;
    const sources = byId("fundamental-source-status");
    sources.replaceChildren();
    const provider = document.createElement("span");
    provider.className = "fundamental-provider is-loading";
    provider.append(document.createElement("b"), document.createTextNode("正在连接基本面数据源"));
    sources.append(provider);
  }

  function renderEvidenceError(message) {
    state.evidenceData = null;
    byId("evidence-meta").textContent = "获取失败";
    byId("evidence-badge").textContent = "证据不可用";
    byId("evidence-score").textContent = "--";
    byId("evidence-label").textContent = "数据源暂不可用";
    byId("evidence-confidence").textContent = "--";
    byId("evidence-coverage").textContent = "--";
    byId("evidence-summary").textContent = message;
    byId("evidence-component-list").replaceChildren();
    renderFundamentals({ warnings: [message] }, { metrics: [], available: false });
  }

  function formatNewsTime(value) {
    if (!value) return "时间未知";
    const timestamp = new Date(value);
    if (Number.isNaN(timestamp.getTime())) return "时间未知";
    return timestamp.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  }

  function setNewsLoading(loading, reset = false) {
    refreshNewsButton.disabled = loading;
    const icon = refreshNewsButton.querySelector("svg");
    if (icon) icon.classList.toggle("spin", loading);
    if (!reset) return;
    byId("news-meta").textContent = "正在获取";
    const sourceStatus = byId("news-source-status");
    sourceStatus.replaceChildren();
    const provider = document.createElement("span");
    provider.className = "news-provider is-loading";
    provider.append(document.createElement("b"), document.createTextNode("正在连接新闻源"));
    sourceStatus.append(provider);
    const list = byId("news-list");
    list.replaceChildren();
    const loadingState = document.createElement("div");
    loadingState.className = "news-empty is-loading";
    loadingState.textContent = "正在获取当前标的的在线新闻";
    list.append(loadingState);
  }

  function renderNewsProviders(providers, warnings, portals = []) {
    const container = byId("news-source-status");
    container.replaceChildren();
    const labels = {
      ok: (provider) => `${safeNumber(provider.item_count)} 条`,
      empty: () => "无结果",
      error: () => "失败",
      timeout: () => "超时",
    };
    (providers || []).forEach((provider) => {
      const status = ["ok", "empty", "error", "timeout"].includes(provider.status)
        ? provider.status
        : "error";
      const sourceUrl = safeExternalUrl(provider.source_url);
      const element = document.createElement(sourceUrl ? "a" : "span");
      element.className = `news-provider is-${status}`;
      element.title = provider.message || `${provider.label} · ${provider.credibility || "在线来源"}`;
      if (sourceUrl) {
        element.href = sourceUrl;
        element.target = "_blank";
        element.rel = "noopener noreferrer";
      }
      const dot = document.createElement("b");
      const statusText = labels[status](provider);
      element.append(dot, document.createTextNode(`${provider.label} · ${provider.credibility || "在线来源"} · ${statusText}`));
      container.append(element);
    });
    if (!container.childElementCount) {
      const unavailable = document.createElement("span");
      unavailable.className = "news-provider is-error";
      unavailable.append(document.createElement("b"), document.createTextNode("在线源不可用"));
      container.append(unavailable);
    }
    const safePortals = (portals || [])
      .map((portal) => ({ ...portal, safeUrl: safeExternalUrl(portal.url) }))
      .filter((portal) => portal.safeUrl);
    if (safePortals.length) {
      const divider = document.createElement("span");
      divider.className = "news-source-divider";
      divider.textContent = "核验入口";
      container.append(divider);
      safePortals.forEach((portal) => {
        const link = document.createElement("a");
        link.className = "news-portal";
        link.href = portal.safeUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = `${portal.label} · ${portal.kind || "原始来源"}`;
        link.append(createIcon("external-link"), document.createTextNode(portal.label));
        container.append(link);
      });
    }
    if (warnings && warnings.length) {
      container.title = warnings.join("；");
    } else {
      container.removeAttribute("title");
    }
  }

  function renderNews(payload) {
    state.newsData = payload;
    const items = (payload.items || [])
      .map((item) => ({ ...item, safeUrl: safeExternalUrl(item.url) }))
      .filter((item) => item.safeUrl);
    byId("news-meta").textContent = `${items.length} 条 · ${formatNewsTime(payload.fetched_at)}`;
    renderNewsProviders(payload.providers || [], payload.warnings || [], payload.source_portals || []);

    const container = byId("news-list");
    container.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "news-empty";
      empty.textContent = (payload.warnings && payload.warnings[0]) || "在线新闻源暂未返回相关内容";
      container.append(empty);
      refreshIcons();
      return;
    }

    const fragment = document.createDocumentFragment();
    items.forEach((item) => {
      const link = document.createElement("a");
      link.className = "news-item";
      link.href = item.safeUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.referrerPolicy = "strict-origin-when-cross-origin";

      const time = document.createElement("time");
      time.dateTime = item.published_at || "";
      time.textContent = formatNewsTime(item.published_at);

      const copy = document.createElement("div");
      copy.className = "news-copy";
      const title = document.createElement("strong");
      title.textContent = item.title;
      const meta = document.createElement("span");
      meta.textContent = [
        item.publisher || "来源未知",
        item.provider_label || item.provider,
        item.credibility,
      ].filter(Boolean).join(" · ");
      copy.append(title, meta);
      if (item.summary) {
        const summary = document.createElement("p");
        summary.textContent = item.summary;
        copy.append(summary);
      }

      const external = createIcon("external-link");
      external.className = "news-external-icon";
      link.append(time, copy, external);
      fragment.append(link);
    });
    container.append(fragment);
    refreshIcons();
  }

  function renderNewsError(message) {
    byId("news-meta").textContent = "获取失败";
    renderNewsProviders([], [message]);
    const container = byId("news-list");
    container.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "news-empty is-error";
    empty.textContent = message;
    container.append(empty);
  }

  async function loadNews(force = false) {
    const requestId = ++state.newsRequestId;
    const newsKey = `${state.market}:${state.symbol}:${state.period}`;
    const reset = force || state.newsKey !== newsKey || !state.newsData || !state.evidenceData;
    state.newsKey = newsKey;
    setNewsLoading(true, reset);
    setEvidenceLoading(true, reset);
    const body = {
      market: state.market,
      symbol: state.symbol,
      period: state.period,
      adjust: "qfq",
      limit: 12,
      force: Boolean(force),
      ...state.quantConfig,
    };
    try {
      const payload = await requestJson("/api/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (requestId !== state.newsRequestId) return;
      renderNews(payload.news || {});
      renderEvidence(payload);
      if (payload.quant_validation && state.data) {
        state.data.quant_validation = payload.quant_validation;
        renderValidation(payload.quant_validation);
      }
    } catch (error) {
      if (requestId !== state.newsRequestId) return;
      state.newsData = null;
      renderNewsError(error.message);
      renderEvidenceError(error.message);
    } finally {
      if (requestId === state.newsRequestId) {
        setNewsLoading(false);
        setEvidenceLoading(false);
      }
    }
  }

  function renderDashboard(data) {
    state.data = data;
    terminal.dataset.market = state.market;
    renderSnapshot(data.snapshot, data.provider);
    renderDataQuality(data.data_quality);
    renderDecision(data.analysis);
    renderFactors(data.analysis.factors || []);
    renderStrategies(data.analysis.strategies || []);
    renderValidation(data.quant_validation);
    renderMining(data.analysis.factor_mining || []);
    renderFactorHeatmap(data.factor_history || []);
    renderKline();
    renderIntraday();
    byId("chart-empty").classList.add("is-hidden");
    const warning = data.warnings && data.warnings.length ? ` · ${data.warnings.join("；")}` : "";
    setStatus("行情已连接", true, `${data.provider.daily_label} · ${formatTime(data.refreshed_at)}${warning}`);
    workspace.setAttribute("aria-busy", "false");
  }

  function mergeLiveCandle(rows) {
    if (!state.data || !rows.length) return;
    const date = rows[rows.length - 1].time.slice(0, 10);
    const prices = rows.map((item) => safeNumber(item.price)).filter((value) => value > 0);
    if (!prices.length) return;
    const candle = {
      date,
      open: prices[0],
      high: Math.max(...prices),
      low: Math.min(...prices),
      close: prices[prices.length - 1],
      volume: safeNumber(rows[rows.length - 1].cumulative_volume) / (state.market === "a-share" ? 100 : 1),
    };
    const candles = state.data.candles;
    if (candles.length && candles[candles.length - 1].date === date) candles[candles.length - 1] = candle;
    else candles.push(candle);
  }

  function applyLiveUpdate(payload) {
    if (!state.data || !payload || !payload.snapshot) return;
    state.data.snapshot = payload.snapshot;
    state.data.intraday = payload.intraday || [];
    mergeLiveCandle(state.data.intraday);
    renderSnapshot(state.data.snapshot, state.data.provider);
    renderKline();
    renderIntraday();
    setStatus("实时更新中", true, `${state.data.snapshot.provider} · ${formatTime(payload.refreshed_at)}`);
  }

  function stopStream() {
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
  }

  function startStream() {
    stopStream();
    if (snapshotMode || state.market === "demo" || !window.EventSource) return;
    const query = new URLSearchParams({ market: state.market, symbol: state.symbol, refresh: "15" });
    if (state.desktopToken) query.set("desktop_token", state.desktopToken);
    const stream = new EventSource(`/api/stream?${query}`);
    state.stream = stream;
    stream.addEventListener("market", (event) => {
      try {
        applyLiveUpdate(JSON.parse(event.data));
      } catch (_error) {
        setStatus("行情解析异常", false, "等待自动重连");
      }
    });
    stream.addEventListener("market-error", (event) => {
      try {
        const payload = JSON.parse(event.data);
        setStatus("实时源暂不可用", false, payload.error || "等待自动重连");
      } catch (_error) {
        setStatus("实时源暂不可用", false, "等待自动重连");
      }
    });
    stream.onerror = () => setStatus("正在重连行情", false, "SSE 连接中断");
  }

  function setMarketSelection(market) {
    document.querySelectorAll(".market-control button").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.market === market);
    });
  }

  function hideInstrumentResults() {
    instrumentResults.classList.add("is-hidden");
    instrumentResults.replaceChildren();
    state.searchIndex = -1;
  }

  function moveSearchSelection(direction) {
    const options = Array.from(instrumentResults.querySelectorAll(".instrument-option"));
    if (!options.length) return;
    state.searchIndex = (state.searchIndex + direction + options.length) % options.length;
    options.forEach((option, index) => {
      const selected = index === state.searchIndex;
      option.classList.toggle("is-selected", selected);
      option.setAttribute("aria-selected", String(selected));
      if (selected) option.scrollIntoView({ block: "nearest" });
    });
  }

  async function selectInstrument(item) {
    const nextMarket = item.market || state.market;
    if (nextMarket !== state.market) {
      state.market = nextMarket;
      terminal.dataset.market = nextMarket;
      setMarketSelection(nextMarket);
      await loadSymbols(nextMarket, "");
    }
    state.symbol = String(item.symbol || "").toUpperCase();
    symbolInput.value = state.symbol;
    hideInstrumentResults();
    await loadDashboard();
  }

  function renderInstrumentResults(payload) {
    instrumentResults.replaceChildren();
    state.searchIndex = -1;
    const items = Array.isArray(payload.items) ? payload.items : [];
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "instrument-empty";
      empty.textContent = payload.warnings && payload.warnings.length
        ? "在线目录暂不可用，可按回车直接加载代码"
        : "未找到匹配证券，可按回车直接加载代码";
      instrumentResults.append(empty);
    }
    items.forEach((item) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "instrument-option";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");

      const identity = document.createElement("span");
      identity.className = "instrument-identity";
      const symbol = document.createElement("strong");
      symbol.textContent = item.symbol;
      const name = document.createElement("span");
      name.textContent = item.name || item.symbol;
      identity.append(symbol, name);

      const metadata = document.createElement("span");
      metadata.className = "instrument-metadata";
      metadata.textContent = [
        item.exchange,
        item.asset_type_label || item.asset_type,
        item.currency,
      ].filter(Boolean).join(" · ");
      option.append(identity, metadata);
      option.addEventListener("click", () => {
        void selectInstrument(item).catch((error) => showToast(error.message));
      });
      instrumentResults.append(option);
    });
    instrumentResults.classList.remove("is-hidden");
  }

  async function searchInstruments() {
    const queryText = symbolInput.value.trim();
    if (!queryText) {
      hideInstrumentResults();
      return;
    }
    if (state.searchController) state.searchController.abort();
    const controller = new AbortController();
    state.searchController = controller;
    const query = new URLSearchParams({
      q: queryText,
      market: state.market,
      asset_type: state.assetType,
      limit: "12",
    });
    try {
      const payload = await requestJson(`/api/instruments/search?${query}`, { signal: controller.signal });
      if (controller !== state.searchController) return;
      renderInstrumentResults(payload);
    } catch (error) {
      if (error.name !== "AbortError" && controller === state.searchController) {
        renderInstrumentResults({ items: [], warnings: [error.message] });
      }
    }
  }

  function scheduleInstrumentSearch() {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => void searchInstruments(), 260);
  }

  async function loadSymbols(market, preferred) {
    const query = new URLSearchParams({ market, asset_type: state.assetType });
    const payload = await requestJson(`/api/symbols?${query}`);
    symbolOptions.replaceChildren();
    payload.symbols.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.symbol;
      option.label = item.label;
      symbolOptions.append(option);
    });
    const preferredSymbol = String(preferred || "").toUpperCase();
    const preferredPreset = payload.symbols.some((item) => item.symbol === preferredSymbol);
    state.symbol = preferredSymbol && (state.assetType === "all" || preferredPreset)
      ? preferredSymbol
      : (payload.symbols[0] ? payload.symbols[0].symbol : payload.definition.default_symbol);
    symbolInput.value = state.symbol;
  }

  function commitSymbol() {
    const candidate = symbolInput.value.trim().toUpperCase();
    if (!candidate || candidate === state.symbol) {
      symbolInput.value = state.symbol;
      hideInstrumentResults();
      return;
    }
    state.symbol = candidate;
    symbolInput.value = candidate;
    hideInstrumentResults();
    loadDashboard();
  }

  function setCatalogMarketSelection(market) {
    document.querySelectorAll("[data-catalog-market]").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.catalogMarket === market);
    });
  }

  function setCatalogBusy(busy) {
    const refreshButton = byId("refresh-catalog-button");
    refreshButton.disabled = busy;
    const icon = refreshButton.querySelector("svg");
    if (icon) icon.classList.toggle("spin", busy);
    byId("catalog-table-body").setAttribute("aria-busy", String(busy));
  }

  function populateCatalogCategories(payload) {
    const selected = state.catalogCategory;
    catalogCategorySelect.replaceChildren();
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = "全部板块";
    catalogCategorySelect.append(all);
    (payload.category_counts || []).forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.category;
      option.textContent = `${entry.category} (${Number(entry.count).toLocaleString("zh-CN")})`;
      catalogCategorySelect.append(option);
    });
    const exists = Array.from(catalogCategorySelect.options).some((option) => option.value === selected);
    if (!exists) state.catalogCategory = "all";
    catalogCategorySelect.value = state.catalogCategory;
  }

  async function chooseCatalogInstrument(item) {
    const nextMarket = item.market || state.catalogMarket;
    instrumentCatalogDialog.close();
    state.market = nextMarket;
    state.assetType = "all";
    terminal.dataset.market = nextMarket;
    assetTypeSelect.value = "all";
    setMarketSelection(nextMarket);
    await loadSymbols(nextMarket, item.symbol);
    state.symbol = String(item.symbol || "").toUpperCase();
    symbolInput.value = state.symbol;
    await loadDashboard();
  }

  function renderCatalog(payload) {
    state.catalogPayload = payload;
    state.catalogPage = Number(payload.page) || 1;
    state.catalogPages = Number(payload.pages) || 1;
    populateCatalogCategories(payload);

    const total = Number(payload.catalog_total) || 0;
    const filtered = Number(payload.filtered_total) || 0;
    byId("catalog-header-count").textContent = `${total.toLocaleString("zh-CN")} 个可选证券`;
    byId("catalog-result-summary").textContent = `当前筛选 ${filtered.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`;
    byId("catalog-page-status").textContent = `第 ${state.catalogPage.toLocaleString("zh-CN")} / ${state.catalogPages.toLocaleString("zh-CN")} 页`;

    const source = payload.source || {};
    const sourceLink = byId("catalog-source-link");
    const sourceUrl = safeExternalUrl(source.url);
    sourceLink.textContent = source.label || "证券目录";
    sourceLink.title = source.warning || source.label || "证券目录";
    if (sourceUrl) sourceLink.href = sourceUrl;
    else sourceLink.removeAttribute("href");
    byId("catalog-updated-at").textContent = source.updated_at
      ? `更新 ${formatTime(source.updated_at)}`
      : "更新时间未知";

    const body = byId("catalog-table-body");
    body.replaceChildren();
    const items = Array.isArray(payload.items) ? payload.items : [];
    items.forEach((item) => {
      const row = document.createElement("tr");
      row.className = "catalog-row";
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-label", `选择 ${item.name || item.symbol} ${item.symbol}`);

      const identityCell = document.createElement("td");
      const identity = document.createElement("div");
      identity.className = "catalog-identity";
      const symbol = document.createElement("strong");
      symbol.textContent = item.symbol;
      const name = document.createElement("span");
      name.textContent = item.name || item.symbol;
      identity.append(symbol, name);
      identityCell.append(identity);

      const exchange = document.createElement("td");
      exchange.textContent = [item.exchange, item.currency].filter(Boolean).join(" · ") || "--";
      const type = document.createElement("td");
      const typeBadge = document.createElement("span");
      typeBadge.className = "catalog-type-badge";
      typeBadge.textContent = item.asset_type_label || item.asset_type || "证券";
      type.append(typeBadge);
      const category = document.createElement("td");
      category.textContent = item.category || "未分类";
      const action = document.createElement("td");
      const actionIcon = document.createElement("span");
      actionIcon.className = "catalog-select-icon";
      actionIcon.append(createIcon("arrow-up-right"));
      action.append(actionIcon);
      row.append(identityCell, exchange, type, category, action);

      const select = () => void chooseCatalogInstrument(item).catch((error) => showToast(error.message));
      row.addEventListener("click", select);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
      body.append(row);
    });
    byId("catalog-empty").classList.toggle("is-hidden", items.length > 0);
    byId("catalog-previous-button").disabled = state.catalogPage <= 1;
    byId("catalog-next-button").disabled = state.catalogPage >= state.catalogPages;
    if (source.warning) showToast(source.warning);
    refreshIcons();
  }

  async function loadCatalog(options = {}) {
    if (state.catalogController) state.catalogController.abort();
    const controller = new AbortController();
    state.catalogController = controller;
    setCatalogBusy(true);
    const query = new URLSearchParams({
      market: state.catalogMarket,
      q: catalogQueryInput.value.trim(),
      asset_type: state.catalogAssetType,
      category: state.catalogCategory,
      page: String(state.catalogPage),
      page_size: "50",
      refresh: String(Boolean(options.refresh)),
    });
    try {
      const payload = await requestJson(`/api/instruments/catalog?${query}`, { signal: controller.signal });
      if (controller !== state.catalogController) return;
      renderCatalog(payload);
    } catch (error) {
      if (error.name === "AbortError" || controller !== state.catalogController) return;
      byId("catalog-table-body").replaceChildren();
      byId("catalog-empty").textContent = `证券目录加载失败：${error.message}`;
      byId("catalog-empty").classList.remove("is-hidden");
      showToast(error.message);
    } finally {
      if (controller === state.catalogController) setCatalogBusy(false);
    }
  }

  function scheduleCatalogSearch() {
    window.clearTimeout(state.catalogTimer);
    state.catalogTimer = window.setTimeout(() => {
      state.catalogPage = 1;
      void loadCatalog();
    }, 260);
  }

  function openInstrumentCatalog() {
    state.catalogMarket = state.market;
    state.catalogAssetType = state.assetType;
    state.catalogCategory = "all";
    state.catalogPage = 1;
    catalogQueryInput.value = "";
    catalogAssetSelect.value = state.catalogAssetType;
    catalogCategorySelect.value = "all";
    setCatalogMarketSelection(state.catalogMarket);
    instrumentCatalogDialog.showModal();
    void loadCatalog();
  }

  async function loadDashboard(options = {}) {
    const requestId = ++state.requestId;
    stopStream();
    setBusy(true);
    setStatus("正在获取行情", false, `${state.market} · ${state.symbol}`);
    const query = new URLSearchParams({
      market: state.market,
      symbol: state.symbol,
      period: state.period,
      adjust: "qfq",
    });
    try {
      const data = await requestJson(`/api/dashboard?${query}`);
      if (requestId !== state.requestId) return;
      renderDashboard(data);
      startStream();
      void loadNews(Boolean(options.forceNews));
    } catch (error) {
      if (requestId !== state.requestId) return;
      setStatus("行情加载失败", false, error.message);
      showToast(error.message);
      byId("chart-empty").classList.remove("is-hidden");
    } finally {
      if (requestId === state.requestId) setBusy(false);
    }
  }

  function writeQuantSettings(config = state.quantConfig) {
    byId("quant-train-rows").value = String(config.train_rows);
    byId("quant-test-rows").value = String(config.test_rows);
    byId("quant-commission-input").value = String(config.commission_bps);
    byId("quant-slippage-input").value = String(config.slippage_bps);
    byId("quant-stress-input").value = String(config.stress_multiplier);
    byId("quant-position-input").value = String(config.max_position_percent);
    byId("quant-position-output").textContent = `${safeNumber(config.max_position_percent).toFixed(0)}%`;
    byId("quant-horizon-select").value = String(config.bootstrap_horizon);
    byId("quant-simulations-select").value = String(config.bootstrap_simulations);
    byId("quant-block-input").value = String(config.bootstrap_block_size);
    byId("quant-seed-input").value = String(config.random_seed);
  }

  function defaultQuantConfig() {
    return {
      train_rows: 0,
      test_rows: 0,
      commission_bps: 3,
      slippage_bps: 2,
      stress_multiplier: 2,
      max_position_percent: 100,
      bootstrap_horizon: 63,
      bootstrap_simulations: 1000,
      bootstrap_block_size: 5,
      random_seed: 7,
    };
  }

  function openQuantSettings() {
    byId("quant-settings-target").textContent = `${state.symbol} · ${marketLabels[state.market] || state.market}`;
    byId("quant-settings-period").textContent = periodSelect.selectedOptions[0]?.textContent || state.period;
    writeQuantSettings();
    if (!quantSettingsDialog.open) quantSettingsDialog.showModal();
  }

  async function applyQuantSettings(event) {
    event.preventDefault();
    if (!quantSettingsForm.reportValidity()) return;
    const next = {
      train_rows: Number(byId("quant-train-rows").value),
      test_rows: Number(byId("quant-test-rows").value),
      commission_bps: Number(byId("quant-commission-input").value),
      slippage_bps: Number(byId("quant-slippage-input").value),
      stress_multiplier: Number(byId("quant-stress-input").value),
      max_position_percent: Number(byId("quant-position-input").value),
      bootstrap_horizon: Number(byId("quant-horizon-select").value),
      bootstrap_simulations: Number(byId("quant-simulations-select").value),
      bootstrap_block_size: Number(byId("quant-block-input").value),
      random_seed: Number(byId("quant-seed-input").value),
    };
    if (next.bootstrap_block_size > next.bootstrap_horizon) {
      showToast("平均区块不能大于风险期限。");
      return;
    }
    state.quantConfig = next;
    quantSettingsDialog.close();
    byId("validation-badge").textContent = "重新计算";
    byId("evidence-badge").textContent = "重新计算";
    showToast("正在按新参数运行本地样本外验证。");
    await loadNews(false);
  }

  const analystNames = {
    market: "技术分析师",
    sentiment: "情绪分析师",
    news: "新闻与宏观分析师",
    fundamentals: "基本面分析师",
  };

  function providerFor(providerId) {
    return state.providers.find((item) => item.id === providerId) || null;
  }

  async function loadResearchProviders() {
    const payload = await requestJson("/api/research/providers");
    state.providers = Array.isArray(payload.providers) ? payload.providers : [];
    const select = byId("provider-select");
    select.replaceChildren();
    state.providers.forEach((provider) => {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = provider.label;
      select.append(option);
    });
    if (!state.providers.length) throw new Error("没有可用的在线模型服务。");
    syncProviderSettings();
  }

  function selectedAnalysts() {
    return Array.from(document.querySelectorAll('input[name="analyst"]:checked')).map((input) => input.value);
  }

  function plannedAgentCount() {
    const debateRounds = Number(byId("debate-rounds-select").value || 1);
    const riskRounds = Number(byId("risk-rounds-select").value || 1);
    return selectedAnalysts().length + debateRounds * 2 + riskRounds * 3 + 3;
  }

  function updateAgentCount() {
    const selected = selectedAnalysts().length;
    const count = plannedAgentCount();
    byId("analyst-selection-count").textContent = `已选择 ${selected} / 4`;
    byId("settings-agent-count").textContent = `${count} 个角色步骤`;
    byId("start-research-button").querySelector("span").textContent = `启动 ${count} 个角色`;
  }

  function setAgentMode(mode) {
    state.agentMode = mode === "online" ? "online" : "offline";
    document.querySelectorAll("[data-agent-mode]").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.agentMode === state.agentMode);
    });
    byId("online-model-settings").classList.toggle("is-hidden", state.agentMode !== "online");
    const fallback = byId("fallback-input");
    const fallbackSetting = byId("fallback-setting");
    fallback.disabled = state.agentMode !== "online";
    fallbackSetting.classList.toggle("is-disabled", fallback.disabled);
    if (fallback.disabled) fallback.checked = true;
    syncProviderSettings();
  }

  function syncProviderSettings() {
    const provider = providerFor(byId("provider-select").value);
    const endpoint = byId("provider-endpoint");
    const modelInput = byId("model-input");
    const keyInput = byId("api-key-input");
    const keyLabel = byId("api-key-label");
    const keyStatus = byId("provider-key-status");
    if (state.agentMode !== "online") {
      keyStatus.textContent = "离线规则无需密钥";
      return;
    }
    if (!provider) {
      endpoint.value = "";
      keyStatus.textContent = "模型服务配置不可用";
      return;
    }
    endpoint.value = provider.base_url || "";
    const previousAutoModel = modelInput.dataset.autoModel || "";
    if (!modelInput.value.trim() || modelInput.value === previousAutoModel) {
      modelInput.value = provider.server_model || "";
      modelInput.dataset.autoModel = provider.server_model || "";
    }
    modelInput.placeholder = provider.id === "ollama" ? "例如 qwen3:8b" : "填写模型 ID";
    keyLabel.textContent = `${provider.label} API Key`;
    keyInput.placeholder = provider.requires_api_key ? "仅用于本次任务" : "可选";
    if (provider.server_key_configured) {
      keyStatus.textContent = `服务端已配置 ${provider.label} 密钥`;
    } else if (provider.requires_api_key) {
      keyStatus.textContent = `需要填写 ${provider.label} API Key`;
    } else {
      keyStatus.textContent = `${provider.label} 本地服务无需密钥`;
    }
  }

  function openResearchSettings() {
    if (state.researchJob && ["queued", "running", "cancelling"].includes(state.researchJob.status)) {
      if (!reportsDialog.open) reportsDialog.showModal();
      return;
    }
    byId("settings-target").textContent = state.symbol;
    byId("settings-market").textContent = marketLabels[state.market] || state.market;
    updateAgentCount();
    if (!agentSettingsDialog.open) agentSettingsDialog.showModal();
  }

  function collectResearchOptions() {
    const analysts = selectedAnalysts();
    if (!analysts.length) throw new Error("请至少选择一名分析师。");
    const provider = providerFor(byId("provider-select").value);
    const model = byId("model-input").value.trim();
    const apiKey = byId("api-key-input").value.trim();
    if (state.agentMode === "online") {
      if (!provider) throw new Error("请选择在线模型服务。");
      if (!model && !provider.server_model) throw new Error("请填写模型 ID。");
      if (provider.requires_api_key && !provider.server_key_configured && !apiKey) {
        throw new Error(`请填写 ${provider.label} API Key。`);
      }
    }
    return {
      market: state.market,
      symbol: state.symbol,
      mode: state.agentMode,
      provider: provider ? provider.id : "openai",
      model,
      api_key: state.agentMode === "online" && apiKey ? apiKey : undefined,
      temperature: Number(byId("temperature-input").value),
      timeout_seconds: Number(byId("timeout-select").value),
      fallback_to_offline: byId("fallback-input").checked,
      fetch_details: byId("fetch-details-input").checked,
      selected_analysts: analysts,
      debate_rounds: Number(byId("debate-rounds-select").value),
      risk_rounds: Number(byId("risk-rounds-select").value),
    };
  }

  function buildAgentPlan(options) {
    const plan = [];
    const add = (agentId, name, reportKey) => {
      plan.push({
        step: plan.length + 1,
        agentId,
        name,
        reportKey,
        status: "waiting",
        message: "等待执行",
        durationMs: 0,
        report: "",
      });
    };
    options.selected_analysts.forEach((agentId) => add(agentId, analystNames[agentId] || agentId, agentId));
    for (let round = 1; round <= options.debate_rounds; round += 1) {
      const suffix = options.debate_rounds > 1 ? `（第 ${round} 轮）` : "";
      add("bull", `看多研究员${suffix}`, options.debate_rounds > 1 ? `bull_${round}` : "bull");
      add("bear", `看空研究员${suffix}`, options.debate_rounds > 1 ? `bear_${round}` : "bear");
    }
    add("research_manager", "研究经理", "research_manager");
    add("trader", "交易员", "trader");
    for (let round = 1; round <= options.risk_rounds; round += 1) {
      const suffix = options.risk_rounds > 1 ? `（第 ${round} 轮）` : "";
      add("risk_aggressive", `激进风险分析师${suffix}`, options.risk_rounds > 1 ? `risk_aggressive_${round}` : "risk_aggressive");
      add("risk_neutral", `中性风险分析师${suffix}`, options.risk_rounds > 1 ? `risk_neutral_${round}` : "risk_neutral");
      add("risk_conservative", `保守风险分析师${suffix}`, options.risk_rounds > 1 ? `risk_conservative_${round}` : "risk_conservative");
    }
    add("portfolio_manager", "组合经理", "portfolio_manager");
    return plan;
  }

  function statusLabel(status) {
    return {
      waiting: "等待执行",
      running: "正在分析",
      completed: "已完成",
      fallback: "离线回退",
      failed: "执行失败",
      cancelled: "已取消",
    }[status] || status;
  }

  function formatDuration(milliseconds) {
    const value = safeNumber(milliseconds);
    if (!value) return "--";
    return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;
  }

  function renderAgentPlan() {
    reportTabs.replaceChildren();
    state.agentPlan.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.id = `agent-step-${item.step}`;
      button.className = "agent-step";
      button.setAttribute("aria-disabled", "true");
      const dot = document.createElement("span");
      dot.className = "agent-step-dot";
      const copy = document.createElement("span");
      copy.className = "agent-step-copy";
      const name = document.createElement("b");
      name.textContent = item.name;
      const status = document.createElement("small");
      status.textContent = item.message;
      copy.append(name, status);
      const duration = document.createElement("span");
      duration.className = "agent-step-time";
      duration.textContent = "--";
      button.append(dot, copy, duration);
      button.addEventListener("click", () => selectAgentReport(item.step));
      reportTabs.append(button);
    });
  }

  function updateAgentStep(item) {
    const button = byId(`agent-step-${item.step}`);
    if (!button) return;
    button.classList.remove("is-running", "is-completed", "is-fallback", "is-failed");
    if (["running", "completed", "fallback", "failed"].includes(item.status)) {
      button.classList.add(`is-${item.status}`);
    }
    button.querySelector("small").textContent = item.message || statusLabel(item.status);
    button.querySelector(".agent-step-time").textContent = formatDuration(item.durationMs);
    button.classList.toggle("has-report", Boolean(item.report));
    button.setAttribute("aria-disabled", String(!item.report));
  }

  function selectAgentReport(step) {
    const item = state.agentPlan[step - 1];
    if (!item || !item.report) return;
    reportTabs.querySelectorAll(".agent-step").forEach((button) => {
      button.classList.toggle("is-selected", button.id === `agent-step-${step}`);
    });
    reportContent.textContent = item.report;
  }

  function setReportMeta(entries) {
    const meta = byId("report-result-meta");
    meta.replaceChildren();
    entries.forEach(([label, value]) => {
      const span = document.createElement("span");
      span.append(document.createTextNode(`${label} `));
      const strong = document.createElement("strong");
      strong.textContent = value;
      span.append(strong);
      meta.append(span);
    });
  }

  function setPipelineProgress(step, total, text) {
    const boundedTotal = Math.max(1, safeNumber(total, state.agentPlan.length || 1));
    const boundedStep = Math.min(boundedTotal, Math.max(0, safeNumber(step)));
    byId("research-progress-bar").style.width = `${(boundedStep / boundedTotal) * 100}%`;
    byId("report-progress-text").textContent = text || `${boundedStep} / ${boundedTotal}`;
  }

  function showReportsDialog() {
    if (agentSettingsDialog.open) agentSettingsDialog.close();
    if (!reportsDialog.open) reportsDialog.showModal();
  }

  function closeResearchStream() {
    if (state.researchStream) {
      state.researchStream.close();
      state.researchStream = null;
    }
  }

  function setResearchBusy(busy) {
    const icon = runAgentsButton.querySelector("svg");
    const label = runAgentsButton.querySelector("span");
    runAgentsButton.disabled = false;
    if (icon) icon.classList.toggle("spin", busy);
    if (label) label.textContent = busy ? "查看进度" : "开始研判";
    byId("cancel-job-button").classList.toggle("is-hidden", !busy);
  }

  function handleProgressEvent(event) {
    const item = state.agentPlan[safeNumber(event.step) - 1];
    if (!item) return;
    item.status = event.status || item.status;
    item.message = event.message || statusLabel(item.status);
    item.durationMs = safeNumber(event.duration_ms);
    updateAgentStep(item);
    const finished = state.agentPlan.filter((agent) => ["completed", "fallback", "failed"].includes(agent.status)).length;
    const displayStep = item.status === "running" ? Math.max(0, safeNumber(event.step) - 1) : safeNumber(event.step);
    setPipelineProgress(displayStep, event.total, `${event.agent_name} · ${item.message} · ${finished}/${event.total}`);
    if (!state.researchResult) {
      reportContent.textContent = `${event.agent_name}\n\n${item.status === "running" ? "正在整合市场证据与上游智能体结论。" : item.message}`;
    }
  }

  function renderReports(payload) {
    state.researchResult = payload;
    const runs = Array.isArray(payload.agent_runs) ? payload.agent_runs : [];
    runs.forEach((run, index) => {
      const item = state.agentPlan[index];
      if (!item) return;
      item.status = run.status || "completed";
      item.message = run.message || statusLabel(item.status);
      item.durationMs = safeNumber(run.duration_ms);
      item.report = (payload.reports && (payload.reports[item.reportKey] || payload.reports[run.agent_id])) || "";
      updateAgentStep(item);
    });
    const decision = payload.decision || {};
    const llm = payload.llm || {};
    const evidence = payload.evidence || {};
    const modelLabel = payload.mode === "online"
      ? `${llm.provider_label || llm.provider || "在线模型"} · ${llm.model || "--"}`
      : "离线可审计规则";
    setReportMeta([
      ["决策", actionLabel(decision.action || "HOLD")],
      ["置信度", `${safeNumber(decision.confidence)}%`],
      ["目标仓位", formatPercent(decision.target_allocation, 0)],
      ["模型", modelLabel],
      ["证据", `${safeNumber(evidence.news_items)} 条新闻 / ${safeNumber(evidence.fundamental_fields)} 项基本面`],
      ["提示", `${(payload.warnings || []).length} 条`],
    ]);
    const preferred = state.agentPlan.find((item) => item.agentId === "portfolio_manager" && item.report)
      || [...state.agentPlan].reverse().find((item) => item.report);
    if (preferred) selectAgentReport(preferred.step);
    else reportContent.textContent = "任务已完成，但没有生成可展示的报告。";

    const block = byId("agent-decision");
    block.classList.add("is-ready");
    block.querySelector("span").textContent = `组合经理：${actionLabel(decision.action)} · 置信度 ${safeNumber(decision.confidence)}% · 仓位 ${formatPercent(decision.target_allocation, 0)}`;
    const fallbackCount = safeNumber(payload.agent_summary && payload.agent_summary.status_counts && payload.agent_summary.status_counts.fallback);
    if (fallbackCount) showToast(`${fallbackCount} 个角色已使用离线规则完成。`);
  }

  async function completeResearchJob() {
    if (!state.researchJob) return;
    try {
      const snapshot = await requestJson(state.researchJob.result_url);
      if (!snapshot.result) throw new Error("任务完成但结果尚未就绪。");
      renderReports(snapshot.result);
      state.researchJob.status = "completed";
      setPipelineProgress(state.agentPlan.length, state.agentPlan.length, "全部智能体已完成");
    } catch (error) {
      reportContent.textContent = `读取研究结果失败：${error.message}`;
      showToast(error.message);
    } finally {
      closeResearchStream();
      setResearchBusy(false);
    }
  }

  function failResearchJob(status, message) {
    closeResearchStream();
    if (state.researchJob) state.researchJob.status = status;
    setResearchBusy(false);
    byId("report-progress-text").textContent = status === "cancelled" ? "任务已取消" : "任务失败";
    reportContent.textContent = message || (status === "cancelled" ? "本次研判已取消。" : "智能体任务执行失败。");
    byId("agent-decision").classList.remove("is-ready");
    byId("agent-decision").querySelector("span").textContent = status === "cancelled" ? "智能体研判已取消" : "智能体研判失败";
    if (message) showToast(message);
  }

  async function handleResearchState(event) {
    const status = event.status || "running";
    if (state.researchJob) state.researchJob.status = status;
    if (status === "running") {
      byId("report-progress-text").textContent = event.message || "智能体任务已启动";
    } else if (status === "cancelling") {
      byId("report-progress-text").textContent = "正在取消当前任务";
      byId("cancel-job-button").disabled = true;
    } else if (status === "completed") {
      await completeResearchJob();
    } else if (status === "failed" || status === "cancelled") {
      failResearchJob(status, event.message || "");
    }
  }

  function connectResearchStream(url) {
    closeResearchStream();
    if (!window.EventSource) {
      failResearchJob("failed", "当前浏览器不支持实时任务进度。");
      return;
    }
    const stream = new EventSource(authenticatedUrl(url));
    state.researchStream = stream;
    stream.addEventListener("progress", (event) => {
      try {
        handleProgressEvent(JSON.parse(event.data));
      } catch (_error) {
        byId("report-progress-text").textContent = "进度数据解析异常";
      }
    });
    stream.addEventListener("state", (event) => {
      try {
        void handleResearchState(JSON.parse(event.data));
      } catch (_error) {
        failResearchJob("failed", "任务状态解析异常。");
      }
    });
    stream.onerror = () => {
      if (state.researchJob && ["queued", "running", "cancelling"].includes(state.researchJob.status)) {
        byId("report-progress-text").textContent = "进度连接中断，正在重连";
      }
    };
  }

  async function startResearch(event) {
    event.preventDefault();
    const startButton = byId("start-research-button");
    let options;
    try {
      options = collectResearchOptions();
    } catch (error) {
      showToast(error.message);
      return;
    }
    startButton.disabled = true;
    runAgentsButton.disabled = true;
    byId("agent-decision").classList.remove("is-ready");
    byId("agent-decision").querySelector("span").textContent = "智能体团队正在研判";
    try {
      const payload = await requestJson("/api/research/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options),
      });
      state.researchJob = { ...payload, status: payload.status || "queued" };
      state.agentPlan = buildAgentPlan(options);
      state.researchResult = null;
      renderAgentPlan();
      reportContent.textContent = "任务已提交，等待第一个智能体开始分析。";
      setReportMeta([["标的", `${state.market} · ${state.symbol}`], ["角色步骤", String(state.agentPlan.length)]]);
      const modeLabel = options.mode === "online"
        ? `${(providerFor(options.provider) || {}).label || options.provider} 在线`
        : "离线规则";
      byId("report-mode-badge").textContent = modeLabel;
      byId("report-mode-badge").classList.toggle("is-online", options.mode === "online");
      byId("cancel-job-button").disabled = false;
      setPipelineProgress(0, payload.total_steps, "任务排队中");
      setResearchBusy(true);
      showReportsDialog();
      connectResearchStream(payload.stream_url);
      byId("api-key-input").value = "";
    } catch (error) {
      runAgentsButton.disabled = false;
      byId("agent-decision").querySelector("span").textContent = "智能体任务未启动";
      showToast(error.message);
    } finally {
      startButton.disabled = false;
    }
  }

  async function cancelResearchJob() {
    if (!state.researchJob || !["queued", "running", "cancelling"].includes(state.researchJob.status)) return;
    const button = byId("cancel-job-button");
    button.disabled = true;
    byId("report-progress-text").textContent = "正在提交取消请求";
    try {
      const payload = await requestJson(state.researchJob.result_url, { method: "DELETE" });
      state.researchJob.status = payload.status || "cancelling";
    } catch (error) {
      button.disabled = false;
      showToast(error.message);
    }
  }

  function runAgents() {
    openResearchSettings();
  }

  function bindEvents() {
    document.querySelectorAll("[data-market]").forEach((button) => {
      if (!button.closest(".market-control")) return;
      button.addEventListener("click", async () => {
        const market = button.dataset.market;
        if (!market || market === state.market) return;
        setMarketSelection(market);
        state.market = market;
        state.assetType = "all";
        assetTypeSelect.value = "all";
        terminal.dataset.market = market;
        if (state.searchController) state.searchController.abort();
        hideInstrumentResults();
        try {
          await loadSymbols(market, marketDefaults[market]);
          await loadDashboard();
        } catch (error) {
          showToast(error.message);
        }
      });
    });

    symbolInput.addEventListener("input", scheduleInstrumentSearch);
    symbolInput.addEventListener("change", () => {
      if (instrumentResults.classList.contains("is-hidden")) commitSymbol();
    });
    symbolInput.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" && !instrumentResults.classList.contains("is-hidden")) {
        event.preventDefault();
        moveSearchSelection(1);
        return;
      }
      if (event.key === "ArrowUp" && !instrumentResults.classList.contains("is-hidden")) {
        event.preventDefault();
        moveSearchSelection(-1);
        return;
      }
      if (event.key === "Escape") {
        hideInstrumentResults();
        return;
      }
      if (event.key !== "Enter") return;
      event.preventDefault();
      const selected = instrumentResults.querySelector(".instrument-option.is-selected");
      if (selected) selected.click();
      else commitSymbol();
    });
    instrumentResults.addEventListener("pointerdown", (event) => event.preventDefault());
    assetTypeSelect.addEventListener("change", async () => {
      state.assetType = assetTypeSelect.value;
      hideInstrumentResults();
      try {
        await loadSymbols(state.market, state.symbol);
        await loadDashboard();
      } catch (error) {
        showToast(error.message);
      }
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".symbol-search")) hideInstrumentResults();
    });
    byId("open-catalog-button").addEventListener("click", openInstrumentCatalog);
    byId("close-catalog-button").addEventListener("click", () => instrumentCatalogDialog.close());
    byId("refresh-catalog-button").addEventListener("click", () => void loadCatalog({ refresh: true }));
    catalogQueryInput.addEventListener("input", scheduleCatalogSearch);
    catalogAssetSelect.addEventListener("change", () => {
      state.catalogAssetType = catalogAssetSelect.value;
      state.catalogCategory = "all";
      state.catalogPage = 1;
      void loadCatalog();
    });
    catalogCategorySelect.addEventListener("change", () => {
      state.catalogCategory = catalogCategorySelect.value;
      state.catalogPage = 1;
      void loadCatalog();
    });
    document.querySelectorAll("[data-catalog-market]").forEach((button) => {
      button.addEventListener("click", () => {
        const market = button.dataset.catalogMarket;
        if (!market || market === state.catalogMarket) return;
        state.catalogMarket = market;
        state.catalogAssetType = "all";
        state.catalogCategory = "all";
        state.catalogPage = 1;
        catalogAssetSelect.value = "all";
        catalogCategorySelect.value = "all";
        setCatalogMarketSelection(market);
        void loadCatalog();
      });
    });
    byId("catalog-previous-button").addEventListener("click", () => {
      if (state.catalogPage <= 1) return;
      state.catalogPage -= 1;
      void loadCatalog();
    });
    byId("catalog-next-button").addEventListener("click", () => {
      if (state.catalogPage >= state.catalogPages) return;
      state.catalogPage += 1;
      void loadCatalog();
    });
    instrumentCatalogDialog.addEventListener("click", (event) => {
      if (event.target === instrumentCatalogDialog) instrumentCatalogDialog.close();
    });
    instrumentCatalogDialog.addEventListener("close", () => {
      window.clearTimeout(state.catalogTimer);
      if (state.catalogController) state.catalogController.abort();
    });
    periodSelect.addEventListener("change", () => {
      state.period = periodSelect.value;
      loadDashboard();
    });
    refreshButton.addEventListener("click", () => loadDashboard({ forceNews: true }));
    refreshNewsButton.addEventListener("click", () => loadNews(true));
    byId("open-quant-settings-button").addEventListener("click", openQuantSettings);
    quantSettingsForm.addEventListener("submit", applyQuantSettings);
    byId("close-quant-settings-button").addEventListener("click", () => quantSettingsDialog.close());
    byId("reset-quant-settings-button").addEventListener("click", () => {
      writeQuantSettings(defaultQuantConfig());
      byId("quant-settings-status").textContent = "默认参数待应用";
    });
    byId("quant-position-input").addEventListener("input", (event) => {
      byId("quant-position-output").textContent = `${safeNumber(event.target.value).toFixed(0)}%`;
    });
    quantSettingsDialog.addEventListener("click", (event) => {
      if (event.target === quantSettingsDialog) quantSettingsDialog.close();
    });
    runAgentsButton.addEventListener("click", runAgents);
    agentSettingsForm.addEventListener("submit", startResearch);
    document.querySelectorAll("[data-agent-mode]").forEach((button) => {
      button.addEventListener("click", () => setAgentMode(button.dataset.agentMode));
    });
    document.querySelectorAll('input[name="analyst"]').forEach((input) => {
      input.addEventListener("change", updateAgentCount);
    });
    byId("debate-rounds-select").addEventListener("change", updateAgentCount);
    byId("risk-rounds-select").addEventListener("change", updateAgentCount);
    byId("provider-select").addEventListener("change", () => {
      byId("api-key-input").value = "";
      syncProviderSettings();
    });
    byId("temperature-input").addEventListener("input", (event) => {
      byId("temperature-output").textContent = Number(event.target.value).toFixed(1);
    });
    byId("toggle-api-key").addEventListener("click", () => {
      const input = byId("api-key-input");
      const icon = byId("toggle-api-key").querySelector("svg");
      const reveal = input.type === "password";
      input.type = reveal ? "text" : "password";
      if (icon) {
        icon.outerHTML = `<i data-lucide="${reveal ? "eye-off" : "eye"}"></i>`;
        if (window.lucide) window.lucide.createIcons();
      }
    });
    byId("close-settings-button").addEventListener("click", () => agentSettingsDialog.close());
    byId("cancel-settings-button").addEventListener("click", () => agentSettingsDialog.close());
    agentSettingsDialog.addEventListener("click", (event) => {
      if (event.target === agentSettingsDialog) agentSettingsDialog.close();
    });
    byId("cancel-job-button").addEventListener("click", cancelResearchJob);
    byId("agent-decision").addEventListener("click", () => {
      if (state.researchResult || state.researchJob) showReportsDialog();
    });

    document.querySelectorAll(".chart-tabs button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".chart-tabs button").forEach((item) => item.classList.toggle("is-selected", item === button));
        const showKline = button.dataset.chart === "kline";
        byId("kline-chart").classList.toggle("is-hidden", !showKline);
        byId("intraday-chart").classList.toggle("is-hidden", showKline);
        window.setTimeout(() => (showKline ? state.charts.kline : state.charts.intraday)?.resize(), 0);
      });
    });

    const navigationTargets = [".market-panel", ".market-panel", ".factor-panel", ".validation-panel", "#run-agents-button", ".decision-panel"];
    document.querySelectorAll(".nav-item").forEach((button, index) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("is-active", item === button));
        if (index === 4) runAgents();
        else document.querySelector(navigationTargets[index])?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    byId("close-dialog-button").addEventListener("click", () => reportsDialog.close());
    reportsDialog.addEventListener("click", (event) => {
      if (event.target === reportsDialog) reportsDialog.close();
    });
    window.addEventListener("beforeunload", () => {
      window.clearTimeout(state.searchTimer);
      window.clearTimeout(state.catalogTimer);
      if (state.searchController) state.searchController.abort();
      if (state.catalogController) state.catalogController.abort();
      stopStream();
      closeResearchStream();
    });
    setAgentMode("offline");
  }

  async function boot() {
    if (window.lucide) window.lucide.createIcons();
    initializeCharts();
    bindEvents();
    try {
      await loadResearchProviders();
      setAgentMode(state.agentMode);
    } catch (error) {
      showToast(`在线模型配置加载失败：${error.message}`);
    }
    try {
      await loadSymbols(state.market, state.symbol);
      await loadDashboard();
    } catch (error) {
      setStatus("初始化失败", false, error.message);
      showToast(error.message);
    }
  }

  boot();
})();
