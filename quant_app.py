from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import (
    BooleanVar,
    END,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = app_root()
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from quant_starter.runtime import ensure_gui_streams

ensure_gui_streams()

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from quant_starter.backtest import BacktestConfig, run_backtest
from quant_starter.data import (
    DemoMarketConfig,
    fetch_a_share_ohlcv,
    fetch_a_share_prices,
    fetch_nasdaq_ohlcv,
    fetch_nasdaq_prices,
    generate_demo_ohlcv,
    generate_demo_prices,
    load_ohlcv_csv,
    load_prices_csv,
    normalize_a_share_symbol,
    parse_symbols,
    provider_display_name,
)
from quant_starter.metrics import format_metrics, summarize_performance
from quant_starter.kline import build_kline_figure, save_kline_chart
from quant_starter.optimization import (
    HoldoutConfig,
    HoldoutValidationResult,
    optimize_risk_managed_momentum,
)
from quant_starter.plots import save_moving_average_chart, save_report_charts
from quant_starter.strategies import (
    RiskManagedMomentumConfig,
    TacticalGrowthConfig,
    momentum_rotation,
    moving_average_crossover,
    moving_average_signals,
    risk_managed_momentum,
)
from quant_starter.symbols import (
    default_stock_choice,
    display_for_symbol,
    resolve_stock_choice,
    stock_choice_labels,
)
from quant_starter.validation import (
    TacticalGrowthValidationResult,
    validate_tactical_growth,
)


DEFAULT_OUTPUT = Path.home() / "Documents" / "QuantStarter" / "outputs"

UI_BG = "#F3F5F7"
UI_SURFACE = "#FFFFFF"
UI_SURFACE_ALT = "#F8FAFB"
UI_BORDER = "#DCE2E7"
UI_TEXT = "#17212B"
UI_MUTED = "#66727F"
UI_TEAL = "#087F82"
UI_WARNING = "#9A5B00"

STRATEGY_LABEL_TO_KEY = {
    "杠杆战术增长（高风险）": "tactical_growth",
    "自适应动量（保留集）": "adaptive",
    "风险调整动量": "risk_momentum",
    "经典动量轮动": "momentum",
    "双均线": "ma",
}
STRATEGY_KEY_TO_LABEL = {
    key: label for label, key in STRATEGY_LABEL_TO_KEY.items()
}


@dataclass(frozen=True)
class KlineRequest:
    source: str
    symbol: str
    start: str
    end: str
    adjust: str
    csv_path: str
    auto_adjust: bool
    seed: int


def run_smoke_test() -> None:
    output_dir = ROOT / "smoke-test-output"
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = generate_demo_ohlcv("ALPHA", DemoMarketConfig(seed=7))
    prices = bars[["Close"]].rename(columns={"Close": "ALPHA"})
    signals = moving_average_signals(prices, "ALPHA", fast=5, slow=20)
    target_weights = moving_average_crossover(prices, "ALPHA", fast=5, slow=20)
    result = run_backtest(prices, target_weights, BacktestConfig())
    metrics = summarize_performance(
        equity=result.equity,
        returns=result.portfolio_returns,
        turnover=result.turnover,
        target_weights=result.target_weights,
    )

    bars.to_csv(output_dir / "ohlcv_data.csv")
    signals.to_csv(output_dir / "moving_average_signals.csv")
    result.equity.rename("equity").to_csv(output_dir / "equity_curve.csv")
    pd.Series(metrics, name="value").to_csv(output_dir / "metrics.csv")
    save_moving_average_chart(result, signals, "ALPHA", output_dir)
    save_kline_chart(bars, "ALPHA", output_dir / "kline_chart.png", window=120)

    required = (
        "ohlcv_data.csv",
        "moving_average_signals.csv",
        "equity_curve.csv",
        "metrics.csv",
        "moving_average_backtest.png",
        "kline_chart.png",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Smoke test output is missing: " + ", ".join(missing))


class QuantStarterApp:
    def __init__(self, root, *, embedded: bool = False) -> None:
        self.root = root
        self.embedded = embedded
        if not embedded:
            self.root.title("量化回测实验室")
            self.root.geometry("1180x800")
            self.root.minsize(960, 700)
            self.root.configure(bg=UI_BG)

        self.source = StringVar(value="nasdaq")
        self.strategy = StringVar(value="tactical_growth")
        self.strategy_label = StringVar(
            value=STRATEGY_KEY_TO_LABEL["tactical_growth"]
        )
        self.strategy_note = StringVar(
            value="高风险：使用每日杠杆 ETF，历史最大回撤接近 50%。"
        )
        self.symbols = StringVar(value="QQQ TQQQ BIL")
        self.stock_choice = StringVar(value="")
        self.start = StringVar(value="2010-03-01")
        self.end = StringVar(value=pd.Timestamp.now().date().isoformat())
        self.adjust = StringVar(value="qfq")
        self.csv_path = StringVar(value="")
        self.output_dir = StringVar(value=str(DEFAULT_OUTPUT))
        self.initial_cash = StringVar(value="100000")
        self.commission_rate = StringVar(value="0.0003")
        self.slippage_rate = StringVar(value="0.0002")
        self.lookback = StringVar(value="126")
        self.rebalance_every = StringVar(value="21")
        self.top_n = StringVar(value="2")
        self.ticker = StringVar(value="ALPHA")
        self.fast = StringVar(value="5")
        self.slow = StringVar(value="20")
        self.skip_recent = StringVar(value="21")
        self.trend_window = StringVar(value="200")
        self.volatility_window = StringVar(value="63")
        self.target_volatility = StringVar(value="0.55")
        self.fast_volatility_window = StringVar(value="21")
        self.volatility_gate = StringVar(value="0.30")
        self.max_position = StringVar(value="1.00")
        self.target_annual_return = StringVar(value="0.20")
        self.max_drawdown_limit = StringVar(value="0.30")
        self.auto_adjust = BooleanVar(value=True)
        self.save_charts = BooleanVar(value=True)
        self.seed = StringVar(value="7")
        self.kline_period = StringVar(value="120")
        self.last_output_dir: Path | None = None
        self.strategy_fields: dict[str, ttk.Entry] = {}

        self.run_button: ttk.Button | None = None
        self.kline_button: ttk.Button | None = None
        self.open_button: ttk.Button | None = None
        self.stock_selector: ttk.Combobox | None = None
        self.results_notebook: ttk.Notebook | None = None
        self.kline_page: ttk.Frame | None = None
        self.kline_container: ttk.Frame | None = None
        self.kline_placeholder: ttk.Label | None = None
        self.kline_canvas: FigureCanvasTkAgg | None = None
        self.kline_figure = None
        self.current_kline_bars: pd.DataFrame | None = None
        self.current_kline_symbol = ""
        self.log: Text | None = None
        self.worker: threading.Thread | None = None
        self.worker_kind: str | None = None

        self._configure_styles()
        self._build_ui()
        self._apply_source_defaults()
        self._set_strategy_field_states()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("QuantPanel.TFrame", background=UI_SURFACE)
        style.configure(
            "QuantPanel.TLabel",
            background=UI_SURFACE,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "QuantSection.TLabel",
            background=UI_SURFACE,
            foreground=UI_TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "QuantWarning.TLabel",
            background=UI_SURFACE,
            foreground=UI_WARNING,
            font=("Microsoft YaHei UI", 8),
        )
        style.configure("QuantPanel.TCheckbutton", background=UI_SURFACE)
        style.map(
            "QuantPanel.TCheckbutton",
            background=[("active", UI_SURFACE)],
        )
        style.configure(
            "QuantPrimary.TButton",
            background=UI_TEAL,
            foreground="#FFFFFF",
            bordercolor=UI_TEAL,
            lightcolor=UI_TEAL,
            darkcolor=UI_TEAL,
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(14, 8),
        )
        style.map(
            "QuantPrimary.TButton",
            background=[("active", "#066E71"), ("disabled", "#9EBFC0")],
        )

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(
            self.root,
            bg=UI_SURFACE,
            highlightbackground=UI_BORDER,
            highlightthickness=1,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = tk.Label(
            header,
            text="策略回测实验室",
            bg=UI_SURFACE,
            fg=UI_TEXT,
            font=("Microsoft YaHei UI", 17, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w", padx=20, pady=(14, 2))
        subtitle = tk.Label(
            header,
            text="A 股 / 纳斯达克 · 战术配置 / 动量 / 双均线 · 成本与滑点",
            bg=UI_SURFACE,
            fg=UI_MUTED,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))

        body = tk.Frame(self.root, bg=UI_BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        data_controls = ttk.Frame(body, style="QuantPanel.TFrame", padding=14)
        data_controls.grid(row=0, column=0, sticky="nsw", padx=(16, 6), pady=16)
        data_controls.columnconfigure(1, weight=1)

        strategy_controls = ttk.Frame(body, style="QuantPanel.TFrame", padding=14)
        strategy_controls.grid(row=0, column=1, sticky="nsw", padx=6, pady=16)
        strategy_controls.columnconfigure(1, weight=1)

        results = ttk.Frame(body, style="QuantPanel.TFrame", padding=16)
        results.grid(row=0, column=2, sticky="nsew", padx=(6, 16), pady=16)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)

        self._add_data_controls(data_controls)
        self._add_cost_controls(data_controls, start_row=10)
        self._add_action_controls(data_controls, start_row=17)
        self._add_strategy_controls(strategy_controls)
        self._add_results(results)

    def _section_label(self, parent: ttk.Frame, text: str, row: int) -> None:
        label = ttk.Label(parent, text=text, style="QuantSection.TLabel")
        label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(16, 6))

    def _entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: StringVar,
        row: int,
        width: int = 24,
    ) -> ttk.Entry:
        field_label = ttk.Label(parent, text=label, style="QuantPanel.TLabel")
        field_label.grid(
            row=row, column=0, sticky="w", pady=3
        )
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        entry.field_label = field_label  # type: ignore[attr-defined]
        return entry

    def _add_data_controls(self, parent: ttk.Frame, *, start_row: int = 0) -> None:
        self._section_label(parent, "行情数据", start_row)
        ttk.Label(parent, text="数据来源", style="QuantPanel.TLabel").grid(
            row=start_row + 1, column=0, sticky="w", pady=3
        )
        source_box = ttk.Combobox(
            parent,
            textvariable=self.source,
            values=("demo", "csv", "a-share", "nasdaq"),
            state="readonly",
            width=20,
        )
        source_box.grid(row=start_row + 1, column=1, columnspan=2, sticky="ew", pady=3)
        source_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_source_defaults())

        ttk.Label(parent, text="选择个股", style="QuantPanel.TLabel").grid(
            row=start_row + 2, column=0, sticky="w", pady=3
        )
        self.stock_selector = ttk.Combobox(
            parent,
            textvariable=self.stock_choice,
            state="normal",
            width=20,
        )
        self.stock_selector.grid(
            row=start_row + 2, column=1, columnspan=2, sticky="ew", pady=3
        )
        self.stock_selector.bind("<<ComboboxSelected>>", self._on_stock_selected)
        self.stock_selector.bind("<Return>", self._on_stock_selected)

        self._entry(parent, "股票列表", self.symbols, start_row + 3)
        self._entry(parent, "开始日期", self.start, start_row + 4)
        self._entry(parent, "结束日期", self.end, start_row + 5)

        ttk.Label(parent, text="A股复权", style="QuantPanel.TLabel").grid(
            row=start_row + 6, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            parent,
            textvariable=self.adjust,
            values=("qfq", "hfq", "none"),
            state="readonly",
            width=20,
        ).grid(row=start_row + 6, column=1, columnspan=2, sticky="ew", pady=3)

        ttk.Label(parent, text="CSV 文件", style="QuantPanel.TLabel").grid(
            row=start_row + 7, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.csv_path, width=24).grid(
            row=start_row + 7, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text="选择", command=self._choose_csv).grid(
            row=start_row + 7, column=2, sticky="e", padx=(6, 0), pady=3
        )

        self._entry(parent, "模拟种子", self.seed, start_row + 8)
        ttk.Checkbutton(
            parent,
            text="美股自动复权",
            variable=self.auto_adjust,
            style="QuantPanel.TCheckbutton",
        ).grid(
            row=start_row + 9,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(6, 2),
        )

    def _add_strategy_controls(self, parent: ttk.Frame, *, start_row: int = 0) -> None:
        self._section_label(parent, "交易策略", start_row)
        ttk.Label(parent, text="策略", style="QuantPanel.TLabel").grid(
            row=start_row + 1, column=0, sticky="w", pady=3
        )
        strategy_box = ttk.Combobox(
            parent,
            textvariable=self.strategy_label,
            values=tuple(STRATEGY_LABEL_TO_KEY),
            state="readonly",
            width=20,
        )
        strategy_box.grid(
            row=start_row + 1, column=1, columnspan=2, sticky="ew", pady=3
        )
        strategy_box.bind("<<ComboboxSelected>>", self._on_strategy_changed)

        ttk.Label(
            parent,
            textvariable=self.strategy_note,
            style="QuantWarning.TLabel",
            wraplength=220,
            justify="left",
        ).grid(
            row=start_row + 2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(2, 5),
        )

        field_specs = (
            ("lookback", "动量周期", self.lookback, 3),
            ("rebalance_every", "调仓间隔", self.rebalance_every, 4),
            ("top_n", "持仓数量", self.top_n, 5),
            ("fast", "短均线", self.fast, 6),
            ("slow", "长均线", self.slow, 7),
            ("skip_recent", "跳过近期", self.skip_recent, 8),
            ("trend_window", "趋势周期", self.trend_window, 9),
            ("volatility_window", "波动周期", self.volatility_window, 10),
            ("target_volatility", "目标波动", self.target_volatility, 11),
            (
                "fast_volatility_window",
                "短波动周期",
                self.fast_volatility_window,
                12,
            ),
            ("volatility_gate", "波动闸门", self.volatility_gate, 13),
            ("max_position", "单股上限", self.max_position, 14),
            (
                "target_annual_return",
                "目标年化",
                self.target_annual_return,
                15,
            ),
            (
                "max_drawdown_limit",
                "回撤上限",
                self.max_drawdown_limit,
                16,
            ),
        )
        for key, label, variable, row_offset in field_specs:
            self.strategy_fields[key] = self._entry(
                parent, label, variable, start_row + row_offset
            )

    def _add_cost_controls(self, parent: ttk.Frame, *, start_row: int = 0) -> None:
        self._section_label(parent, "资金与成本", start_row)
        self._entry(parent, "初始资金", self.initial_cash, start_row + 1)
        self._entry(parent, "手续费率", self.commission_rate, start_row + 2)
        self._entry(parent, "滑点率", self.slippage_rate, start_row + 3)

        ttk.Label(parent, text="输出目录", style="QuantPanel.TLabel").grid(
            row=start_row + 4, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.output_dir, width=24).grid(
            row=start_row + 4, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text="选择", command=self._choose_output_dir).grid(
            row=start_row + 4, column=2, sticky="e", padx=(6, 0), pady=3
        )
        ttk.Checkbutton(
            parent,
            text="保存图表",
            variable=self.save_charts,
            style="QuantPanel.TCheckbutton",
        ).grid(
            row=start_row + 5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(6, 2),
        )

    def _add_action_controls(self, parent: ttk.Frame, *, start_row: int = 0) -> None:
        actions = ttk.Frame(parent, style="QuantPanel.TFrame")
        actions.grid(
            row=start_row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(18, 0),
        )
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)

        self.kline_button = ttk.Button(
            actions,
            text="查看K线",
            command=self._start_kline_preview,
        )
        self.kline_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.run_button = ttk.Button(
            actions,
            text="运行回测",
            command=self._run_async,
            style="QuantPrimary.TButton",
        )
        self.run_button.grid(row=0, column=1, sticky="ew", padx=4)

        self.open_button = ttk.Button(
            actions,
            text="打开结果",
            command=self._open_output_dir,
            state="disabled",
        )
        self.open_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _add_results(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="运行结果", style="QuantSection.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.results_notebook = ttk.Notebook(parent)
        self.results_notebook.grid(row=1, column=0, columnspan=2, sticky="nsew")

        summary_page = ttk.Frame(self.results_notebook, padding=0)
        summary_page.columnconfigure(0, weight=1)
        summary_page.rowconfigure(0, weight=1)
        self.results_notebook.add(summary_page, text="回测摘要")

        self.log = Text(
            summary_page,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            height=20,
            bg=UI_SURFACE_ALT,
            fg=UI_TEXT,
            insertbackground=UI_TEXT,
            selectbackground="#DDF1F0",
            borderwidth=0,
            relief="flat",
            padx=14,
            pady=12,
        )
        self.log.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(summary_page, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        self.kline_page = ttk.Frame(self.results_notebook, padding=10)
        self.kline_page.columnconfigure(0, weight=1)
        self.kline_page.rowconfigure(1, weight=1)
        self.results_notebook.add(self.kline_page, text="个股K线")

        toolbar = ttk.Frame(self.kline_page)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="显示周期").pack(side="left")
        ttk.Combobox(
            toolbar,
            textvariable=self.kline_period,
            values=("60", "120", "250", "500", "全部"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(8, 6))
        ttk.Button(
            toolbar,
            text="刷新",
            command=self._refresh_kline_chart,
        ).pack(side="left")

        self.kline_container = ttk.Frame(self.kline_page)
        self.kline_container.grid(row=1, column=0, sticky="nsew")
        self.kline_container.columnconfigure(0, weight=1)
        self.kline_container.rowconfigure(0, weight=1)
        self.kline_placeholder = ttk.Label(
            self.kline_container,
            text="尚未载入K线数据",
            anchor="center",
        )
        self.kline_placeholder.grid(row=0, column=0, sticky="nsew")
        self._write_log("准备就绪。选择数据来源，然后点击“运行回测”。")

    def _choose_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="选择行情 CSV",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if path:
            self.csv_path.set(path)
            self.source.set("csv")
            self._refresh_stock_choices(reset=False)

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir.set(path)

    def _open_output_dir(self) -> None:
        target = self.last_output_dir or Path(self.output_dir.get()).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)

    def _on_strategy_changed(self, _event=None) -> None:
        key = STRATEGY_LABEL_TO_KEY[self.strategy_label.get()]
        self.strategy.set(key)
        if key == "tactical_growth":
            self.strategy_note.set(
                "高风险：使用每日杠杆 ETF，历史最大回撤接近 50%。"
            )
            self.source.set("nasdaq")
            self.symbols.set("QQQ TQQQ BIL")
            self.ticker.set("QQQ")
            self.start.set("2010-03-01")
            self.end.set(pd.Timestamp.now().date().isoformat())
            self.trend_window.set("200")
            self.volatility_window.set("63")
            self.target_volatility.set("0.55")
            self.fast_volatility_window.set("21")
            self.volatility_gate.set("0.30")
            self.max_position.set("1.00")
        else:
            self.strategy_note.set("")
        self._refresh_stock_choices(reset=True)
        self._set_strategy_field_states()

    def _set_strategy_field_states(self) -> None:
        enabled_by_strategy = {
            "tactical_growth": {
                "trend_window",
                "volatility_window",
                "target_volatility",
                "fast_volatility_window",
                "volatility_gate",
                "max_position",
                "target_annual_return",
            },
            "adaptive": {"target_annual_return", "max_drawdown_limit"},
            "risk_momentum": {
                "lookback",
                "rebalance_every",
                "top_n",
                "skip_recent",
                "trend_window",
                "volatility_window",
                "target_volatility",
                "max_position",
            },
            "momentum": {"lookback", "rebalance_every", "top_n"},
            "ma": {"fast", "slow"},
        }
        enabled = enabled_by_strategy[self.strategy.get()]
        for key, entry in self.strategy_fields.items():
            field_label = entry.field_label  # type: ignore[attr-defined]
            if key in enabled:
                field_label.grid()
                entry.grid()
                entry.configure(state="normal")
            else:
                field_label.grid_remove()
                entry.grid_remove()

    def _apply_source_defaults(self) -> None:
        if self.source.get() == "a-share":
            self.symbols.set("600519 000001 300750")
            self.ticker.set("600519")
        elif self.source.get() == "nasdaq":
            if self.strategy.get() == "tactical_growth":
                self.symbols.set("QQQ TQQQ BIL")
                self.ticker.set("QQQ")
            else:
                self.symbols.set("AAPL MSFT NVDA")
                self.ticker.set("AAPL")
        elif self.source.get() == "demo":
            self.symbols.set("ALPHA BALANCE CYCLE DEFENSE GROWTH VALUE")
            self.ticker.set("ALPHA")
        self._refresh_stock_choices(reset=True)

    def _refresh_stock_choices(self, *, reset: bool) -> None:
        source = self.source.get()
        if self.stock_selector is not None:
            self.stock_selector.configure(values=stock_choice_labels(source))

        symbol = self.ticker.get().strip()
        if not reset and self.stock_choice.get().strip():
            try:
                symbol = resolve_stock_choice(self.stock_choice.get(), source)
            except ValueError:
                pass
        if not symbol:
            symbol = resolve_stock_choice(default_stock_choice(source), source)
        self.ticker.set(symbol)
        self.stock_choice.set(display_for_symbol(source, symbol))
        self._clear_kline()

    def _on_stock_selected(self, _event=None) -> None:
        try:
            symbol = resolve_stock_choice(self.stock_choice.get(), self.source.get())
        except ValueError as exc:
            messagebox.showerror("选择个股", str(exc))
            return
        self.ticker.set(symbol)
        self.stock_choice.set(display_for_symbol(self.source.get(), symbol))
        if self.strategy.get() == "ma":
            self.symbols.set(symbol)
        self._clear_kline()

    def _set_running(self, running: bool) -> None:
        if self.run_button is not None:
            self.run_button.configure(state="disabled" if running else "normal")
        if self.kline_button is not None:
            self.kline_button.configure(state="disabled" if running else "normal")

    def _write_log(self, text: str) -> None:
        if self.log is None:
            return
        self.log.insert(END, text + "\n")
        self.log.see(END)

    def _replace_log(self, text: str) -> None:
        if self.log is None:
            return
        self.log.delete("1.0", END)
        self._write_log(text)

    def _run_async(self) -> None:
        if self.worker_kind is not None or (
            self.worker is not None and self.worker.is_alive()
        ):
            return
        try:
            self._selected_ticker()
        except ValueError as exc:
            messagebox.showerror("量化回测入门", str(exc))
            return
        self._set_running(True)
        self.worker_kind = "backtest"
        self._replace_log("正在获取数据并运行回测，请稍候...")
        self.worker = threading.Thread(target=self._run_backtest, daemon=True)
        self.worker.start()

    def _selected_ticker(self) -> str:
        raw_choice = self.stock_choice.get().strip()
        if raw_choice:
            ticker = resolve_stock_choice(raw_choice, self.source.get())
        else:
            raw_ticker = self.ticker.get().strip()
            if not raw_ticker:
                raw_ticker = parse_symbols(self.symbols.get())[0]
            ticker = raw_ticker.upper()
        if self.source.get() == "a-share":
            ticker = normalize_a_share_symbol(ticker)
        return ticker

    def _snapshot_kline_request(self) -> KlineRequest:
        return KlineRequest(
            source=self.source.get(),
            symbol=self._selected_ticker(),
            start=self.start.get().strip(),
            end=self.end.get().strip(),
            adjust=self.adjust.get(),
            csv_path=self.csv_path.get().strip(),
            auto_adjust=bool(self.auto_adjust.get()),
            seed=int(self.seed.get()),
        )

    @staticmethod
    def _load_selected_bars(request: KlineRequest) -> pd.DataFrame:
        if request.source == "demo":
            return generate_demo_ohlcv(
                request.symbol,
                DemoMarketConfig(seed=request.seed),
            )
        if request.source == "csv":
            if not request.csv_path:
                raise ValueError("请先选择包含 OHLCV 的 CSV 文件。")
            return load_ohlcv_csv(request.csv_path)
        if request.source == "a-share":
            return fetch_a_share_ohlcv(
                request.symbol,
                start=request.start,
                end=request.end,
                adjust=request.adjust,
            )
        if request.source == "nasdaq":
            return fetch_nasdaq_ohlcv(
                request.symbol,
                start=request.start,
                end=request.end,
                auto_adjust=request.auto_adjust,
            )
        raise ValueError(f"未知数据来源：{request.source}")

    def _start_kline_preview(self) -> None:
        if self.worker_kind is not None or (
            self.worker is not None and self.worker.is_alive()
        ):
            return
        try:
            request = self._snapshot_kline_request()
        except (ValueError, IndexError) as exc:
            messagebox.showerror("个股K线", str(exc))
            return

        self._set_running(True)
        self.worker_kind = "kline"
        if self.results_notebook is not None and self.kline_page is not None:
            self.results_notebook.select(self.kline_page)
        if self.kline_placeholder is not None:
            self.kline_placeholder.configure(text=f"正在载入 {request.symbol} ...")
            self.kline_placeholder.grid()
        self.worker = threading.Thread(
            target=self._kline_worker,
            args=(request,),
            daemon=True,
        )
        self.worker.start()

    def _kline_worker(self, request: KlineRequest) -> None:
        try:
            bars = self._load_selected_bars(request)
            self.root.after(
                0,
                lambda data=bars, symbol=request.symbol: self._on_kline_success(
                    data, symbol
                ),
            )
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._on_error(error))

    def _selected_kline_window(self) -> int | None:
        value = self.kline_period.get()
        return None if value == "全部" else int(value)

    def _refresh_kline_chart(self) -> None:
        if self.current_kline_bars is None or not self.current_kline_symbol:
            return
        try:
            self._render_kline_chart(
                self.current_kline_bars,
                self.current_kline_symbol,
            )
        except Exception as exc:
            messagebox.showerror("个股K线", str(exc))

    def _render_kline_chart(self, bars: pd.DataFrame, symbol: str) -> None:
        if self.kline_container is None:
            return
        self._clear_kline(reset_data=False)
        figure = build_kline_figure(
            bars,
            symbol,
            window=self._selected_kline_window(),
        )
        self.kline_figure = figure
        self.kline_canvas = FigureCanvasTkAgg(figure, master=self.kline_container)
        self.kline_canvas.draw()
        if self.kline_placeholder is not None:
            self.kline_placeholder.grid_remove()
        self.kline_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.current_kline_bars = bars
        self.current_kline_symbol = symbol

    def _clear_kline(self, *, reset_data: bool = True) -> None:
        if self.kline_canvas is not None:
            self.kline_canvas.get_tk_widget().destroy()
            self.kline_canvas = None
        if self.kline_figure is not None:
            self.kline_figure.clear()
            self.kline_figure = None
        if reset_data:
            self.current_kline_bars = None
            self.current_kline_symbol = ""
        if self.kline_placeholder is not None:
            self.kline_placeholder.configure(text="尚未载入K线数据")
            self.kline_placeholder.grid()

    def _on_kline_success(self, bars: pd.DataFrame, symbol: str) -> None:
        try:
            self._render_kline_chart(bars, symbol)
        except Exception as exc:
            self._on_error(exc)
            return
        if self.results_notebook is not None and self.kline_page is not None:
            self.results_notebook.select(self.kline_page)
        provider = provider_display_name(bars.attrs.get("provider", "demo"))
        self._write_log(
            f"K线已载入：{symbol} · {len(bars):,} 条 · 数据源 {provider}"
        )
        self.worker_kind = None
        self._set_running(False)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        source = self.source.get()
        if self.strategy.get() == "tactical_growth" and source not in {
            "nasdaq",
            "csv",
        }:
            raise ValueError(
                "杠杆战术增长仅支持纳斯达克实时数据或包含 "
                "QQQ、TQQQ、BIL 的 CSV。"
            )
        if source == "demo":
            config = DemoMarketConfig(seed=int(self.seed.get()))
            prices = generate_demo_prices(config)
            bars = None
            if self.strategy.get() == "ma":
                bars = generate_demo_ohlcv(self._selected_ticker(), config)
            return prices, bars
        if source == "csv":
            if not self.csv_path.get().strip():
                raise ValueError("Choose a CSV file first.")
            csv_path = self.csv_path.get()
            header = pd.read_csv(csv_path, nrows=0)
            header_names = {str(column).strip().lower() for column in header.columns}
            if {"open", "high", "low", "close"}.issubset(header_names):
                bars = load_ohlcv_csv(csv_path)
                ticker = self._selected_ticker()
                prices = bars[["Close"]].rename(columns={"Close": ticker})
                return prices, bars
            return load_prices_csv(csv_path), None
        if source == "a-share":
            if self.strategy.get() == "ma":
                ticker = self._selected_ticker()
                bars = fetch_a_share_ohlcv(
                    ticker,
                    start=self.start.get(),
                    end=self.end.get(),
                    adjust=self.adjust.get(),
                )
                return bars[["Close"]].rename(columns={"Close": ticker}), bars
            return fetch_a_share_prices(
                symbols=parse_symbols(self.symbols.get()),
                start=self.start.get(),
                end=self.end.get(),
                adjust=self.adjust.get(),
            ), None
        if source == "nasdaq":
            if self.strategy.get() == "ma":
                ticker = self._selected_ticker()
                bars = fetch_nasdaq_ohlcv(
                    ticker,
                    start=self.start.get(),
                    end=self.end.get(),
                    auto_adjust=self.auto_adjust.get(),
                )
                return bars[["Close"]].rename(columns={"Close": ticker}), bars
            return fetch_nasdaq_prices(
                symbols=parse_symbols(self.symbols.get()),
                start=self.start.get(),
                end=self.end.get(),
                auto_adjust=self.auto_adjust.get(),
            ), None
        raise ValueError(f"Unknown data source: {source}")

    def _backtest_config(self) -> BacktestConfig:
        return BacktestConfig(
            initial_cash=float(self.initial_cash.get()),
            commission_rate=float(self.commission_rate.get()),
            slippage_rate=float(self.slippage_rate.get()),
            benchmark_symbol=(
                "QQQ" if self.strategy.get() == "tactical_growth" else None
            ),
        )

    def _risk_momentum_config(self) -> RiskManagedMomentumConfig:
        return RiskManagedMomentumConfig(
            lookback=int(self.lookback.get()),
            skip_recent=int(self.skip_recent.get()),
            trend_window=int(self.trend_window.get()),
            volatility_window=int(self.volatility_window.get()),
            rebalance_every=int(self.rebalance_every.get()),
            top_n=int(self.top_n.get()),
            target_volatility=float(self.target_volatility.get()),
            max_position=float(self.max_position.get()),
        )

    def _tactical_growth_config(self) -> TacticalGrowthConfig:
        return TacticalGrowthConfig(
            trend_window=int(self.trend_window.get()),
            volatility_window=int(self.volatility_window.get()),
            fast_volatility_window=int(self.fast_volatility_window.get()),
            target_volatility=float(self.target_volatility.get()),
            volatility_gate=float(self.volatility_gate.get()),
            max_growth_weight=float(self.max_position.get()),
        )

    def _build_weights(
        self,
        prices: pd.DataFrame,
        backtest_config: BacktestConfig,
    ) -> tuple[
        pd.DataFrame,
        HoldoutValidationResult | TacticalGrowthValidationResult | None,
    ]:
        if self.strategy.get() == "tactical_growth":
            validation = validate_tactical_growth(
                prices,
                config=self._tactical_growth_config(),
                backtest=backtest_config,
                target_annual_return=float(self.target_annual_return.get()),
            )
            return validation.target_weights, validation
        if self.strategy.get() == "adaptive":
            validation = optimize_risk_managed_momentum(
                prices,
                holdout=HoldoutConfig(
                    target_annual_return=float(self.target_annual_return.get()),
                    max_drawdown_limit=float(self.max_drawdown_limit.get()),
                ),
                backtest=backtest_config,
            )
            return validation.target_weights, validation
        if self.strategy.get() == "risk_momentum":
            return risk_managed_momentum(
                prices, self._risk_momentum_config()
            ), None
        if self.strategy.get() == "momentum":
            return (
                momentum_rotation(
                    prices,
                    lookback=int(self.lookback.get()),
                    rebalance_every=int(self.rebalance_every.get()),
                    top_n=int(self.top_n.get()),
                ),
                None,
            )
        return (
            moving_average_crossover(
                prices,
                ticker=self._selected_ticker(),
                fast=int(self.fast.get()),
                slow=int(self.slow.get()),
            ),
            None,
        )

    def _run_backtest(self) -> None:
        try:
            output_dir = Path(self.output_dir.get()).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            prices, bars = self._load_market_data()
            backtest_config = self._backtest_config()
            target_weights, validation = self._build_weights(
                prices, backtest_config
            )
            result = run_backtest(
                prices,
                target_weights,
                backtest_config,
            )
            metrics = summarize_performance(
                equity=result.equity,
                returns=result.portfolio_returns,
                turnover=result.turnover,
                target_weights=result.target_weights,
            )

            prices.to_csv(output_dir / "prices.csv")
            result.target_weights.to_csv(output_dir / "target_weights.csv")
            result.equity.rename("equity").to_csv(output_dir / "equity_curve.csv")
            result.turnover.rename("turnover").to_csv(output_dir / "turnover.csv")
            pd.Series(metrics, name="value").to_csv(output_dir / "metrics.csv")
            if isinstance(validation, HoldoutValidationResult):
                validation.leaderboard.to_csv(
                    output_dir / "parameter_leaderboard.csv", index=False
                )
                (output_dir / "holdout_validation.json").write_text(
                    json.dumps(
                        validation.summary_dict(),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            elif isinstance(validation, TacticalGrowthValidationResult):
                validation.diagnostics.to_csv(
                    output_dir / "tactical_growth_diagnostics.csv"
                )
                transitions = validation.diagnostics["RiskOn"].ne(
                    validation.diagnostics["RiskOn"].shift()
                )
                transitions.iloc[0] = False
                validation.diagnostics.loc[transitions].to_csv(
                    output_dir / "trades.csv"
                )
                (output_dir / "tactical_growth_validation.json").write_text(
                    json.dumps(
                        validation.summary_dict(),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            if bars is not None:
                bars.to_csv(output_dir / "ohlcv_data.csv")

            signals = None
            if self.strategy.get() == "ma":
                ticker = self._selected_ticker()
                signals = moving_average_signals(
                    prices,
                    ticker,
                    fast=int(self.fast.get()),
                    slow=int(self.slow.get()),
                )
                signals.to_csv(output_dir / "moving_average_signals.csv")
                signals.loc[signals["Signal"] != 0].to_csv(
                    output_dir / "trades.csv"
                )

            chart_paths: list[Path] = []
            if self.save_charts.get():
                chart_paths = save_report_charts(result, output_dir)
                if bars is not None:
                    chart_paths.append(
                        save_kline_chart(
                            bars,
                            self._selected_ticker(),
                            output_dir / "kline_chart.png",
                            window=250,
                        )
                    )
                if signals is not None:
                    chart_paths.append(
                        save_moving_average_chart(
                            result,
                            signals,
                            self._selected_ticker(),
                            output_dir,
                        )
                    )

            lines = [
                "回测完成",
                f"数据来源: {self.source.get()}",
                f"策略: {STRATEGY_KEY_TO_LABEL[self.strategy.get()]}",
                f"数据行数: {len(prices):,}",
                f"日期范围: {prices.index.min().date()} 至 {prices.index.max().date()}",
                f"输出目录: {output_dir.resolve()}",
                "",
                "关键指标",
                format_metrics(metrics),
            ]
            providers = prices.attrs.get("providers", {})
            if providers:
                lines[5:5] = [
                    "数据提供商: "
                    + ", ".join(
                        f"{symbol}={provider_display_name(provider)}"
                        for symbol, provider in providers.items()
                    )
                ]
            if signals is not None:
                lines.extend(
                    [
                        "",
                        f"金叉/死叉次数: {(signals['Signal'] != 0).sum()}",
                    ]
                )
            if isinstance(validation, HoldoutValidationResult):
                selected = validation.selected
                lines.extend(
                    [
                        "",
                        "训练 / 保留集验证",
                        f"切分日期: {validation.split_date}",
                        (
                            "训练集目标: "
                            + ("达成" if validation.train_target_met else "未达成")
                        ),
                        (
                            "保留集目标: "
                            + ("达成" if validation.test_target_met else "未达成")
                        ),
                        (
                            "保留集回撤约束: "
                            + ("通过" if validation.test_risk_limit_met else "未通过")
                        ),
                        (
                            "选中参数: "
                            f"lookback={selected.lookback}, "
                            f"trend={selected.trend_window}, "
                            f"rebalance={selected.rebalance_every}, "
                            f"top_n={selected.top_n}"
                        ),
                        "",
                        "训练集指标",
                        format_metrics(validation.train_metrics),
                        "",
                        "保留集指标",
                        format_metrics(validation.test_metrics),
                    ]
                )
            elif isinstance(validation, TacticalGrowthValidationResult):
                full = validation.full_period
                holdout = validation.holdout_period
                block_returns = " / ".join(
                    f"{period.metrics['annual_return']:.2%}"
                    for period in validation.chronological_blocks
                )
                lines.extend(
                    [
                        "",
                        "杠杆战术增长 · 固定规则时间顺序验证",
                        (
                            f"历史目标 {validation.target_annual_return:.0%}: "
                            + (
                                "达成"
                                if validation.historical_target_met
                                else "未达成"
                            )
                        ),
                        (
                            f"成本翻倍（单边 {validation.stress_one_way_cost:.2%}）: "
                            + (
                                "通过"
                                if validation.cost_stress_target_met
                                else "未通过"
                            )
                        ),
                        f"验证起点: {full.start_date}",
                        (
                            "完整验证段: "
                            f"年化 {full.metrics['annual_return']:.2%} / "
                            f"回撤 {full.metrics['max_drawdown']:.2%}"
                        ),
                        (
                            "后 30% 保留段: "
                            f"{holdout.start_date} 至 {holdout.end_date} / "
                            f"年化 {holdout.metrics['annual_return']:.2%} / "
                            f"回撤 {holdout.metrics['max_drawdown']:.2%}"
                        ),
                        (
                            "成本翻倍保留段年化: "
                            f"{validation.stress_holdout_period.metrics['annual_return']:.2%}"
                        ),
                        f"四个连续区段年化: {block_returns}",
                        "",
                        "风险提示",
                        "TQQQ 是每日 3 倍杠杆 ETF；历史结果不代表未来，"
                        "本金可能出现接近或超过 50% 的回撤。",
                    ]
                )
            if bars is not None:
                lines.extend(["", "OHLCV 前 5 行", bars.head().to_string()])
            if chart_paths:
                lines.extend(["", "图表"])
                lines.extend(f"- {path.resolve()}" for path in chart_paths)

            self.last_output_dir = output_dir
            preview_symbol = self._selected_ticker() if bars is not None else None
            self.root.after(
                0,
                lambda text="\n".join(lines), data=bars, symbol=preview_symbol: (
                    self._on_success(text, data, symbol)
                ),
            )
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._on_error(error))

    def _on_success(
        self,
        text: str,
        bars: pd.DataFrame | None = None,
        symbol: str | None = None,
    ) -> None:
        self._replace_log(text)
        if bars is not None and symbol is not None:
            try:
                self._render_kline_chart(bars, symbol)
            except Exception as exc:
                self._write_log("\nK线显示失败：" + str(exc))
        if self.results_notebook is not None:
            self.results_notebook.select(0)
        if self.open_button is not None:
            self.open_button.configure(state="normal")
        self.worker_kind = None
        self._set_running(False)

    def _on_error(self, exc: Exception) -> None:
        message = str(exc)
        self._replace_log("发生错误:\n" + message)
        messagebox.showerror("量化回测入门", message)
        self.worker_kind = None
        self._set_running(False)


def main() -> None:
    if "--smoke-test" in sys.argv:
        run_smoke_test()
        return

    root = Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    QuantStarterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
