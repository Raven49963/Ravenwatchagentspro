from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import (
    BooleanVar,
    DoubleVar,
    END,
    IntVar,
    StringVar,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)
from tkinter.scrolledtext import ScrolledText

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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from quant_app import QuantStarterApp
from quant_starter.agent_workflow import (
    ProgressEvent,
    ResearchResult,
    RavenWatchAgentsWorkflow,
    WorkflowCancelled,
    WorkflowConfig,
)
from quant_starter.llm_client import LLMSettings, OpenAICompatibleClient
from quant_starter.kline import build_kline_figure
from quant_starter.research_data import ResearchContext, build_research_context
from quant_starter.research_store import (
    append_decision_memory,
    load_memory_context,
    save_research_result,
)
from quant_starter.symbols import (
    default_stock_choice,
    display_for_symbol,
    resolve_stock_choice,
    stock_choice_labels,
)


APP_HOME = Path.home() / "Documents" / "RavenWatchAgentsCN"
DEFAULT_OUTPUT = APP_HOME / "research"
DEFAULT_MEMORY = APP_HOME / "memory" / "decision_log.jsonl"


COLORS = {
    "bg": "#F3F5F7",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFB",
    "sidebar": "#20262D",
    "sidebar_hover": "#2C343D",
    "sidebar_active": "#126E72",
    "text": "#17212B",
    "muted": "#66727F",
    "muted_light": "#9AA4AE",
    "border": "#DCE2E7",
    "teal": "#087F82",
    "teal_soft": "#DDF1F0",
    "green": "#197149",
    "green_soft": "#E3F3EA",
    "amber": "#956700",
    "amber_soft": "#F8EBCB",
    "red": "#B33A46",
    "red_soft": "#F8E3E5",
    "blue": "#2C65A7",
    "blue_soft": "#E5EEF8",
}


PIPELINE_GROUPS = (
    (
        "市场分析",
        (
            ("market", "技术"),
            ("sentiment", "情绪"),
            ("news", "新闻"),
            ("fundamentals", "基本面"),
        ),
    ),
    (
        "观点形成",
        (
            ("bull", "看多研究"),
            ("bear", "看空研究"),
            ("research_manager", "研究经理"),
        ),
    ),
    (
        "交易与风控",
        (
            ("trader", "交易员"),
            ("risk_aggressive", "激进风险"),
            ("risk_neutral", "中性风险"),
            ("risk_conservative", "保守风险"),
        ),
    ),
    ("组合决策", (("portfolio_manager", "组合经理"),)),
)


class AgentPipelineView(tk.Frame):
    """Compact, stateful view of the Raven Watch Agents collaboration graph."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg=COLORS["surface"])
        self.rows: dict[str, tuple[tk.Label, tk.Label]] = {}
        for column, (group_name, agents) in enumerate(PIPELINE_GROUPS):
            self.columnconfigure(column, weight=1, uniform="pipeline")
            group = tk.Frame(self, bg=COLORS["surface"])
            group.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 8, 0),
            )
            tk.Label(
                group,
                text=group_name,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(0, 8))
            for agent_id, agent_name in agents:
                row = tk.Frame(group, bg=COLORS["surface_alt"], height=29)
                row.pack(fill="x", pady=2)
                row.pack_propagate(False)
                dot = tk.Label(
                    row,
                    text="●",
                    bg=COLORS["surface_alt"],
                    fg=COLORS["muted_light"],
                    font=("Segoe UI Symbol", 8),
                    width=2,
                )
                dot.pack(side="left", padx=(5, 0))
                tk.Label(
                    row,
                    text=agent_name,
                    bg=COLORS["surface_alt"],
                    fg=COLORS["text"],
                    font=("Microsoft YaHei UI", 9),
                    anchor="w",
                ).pack(side="left", fill="x", expand=True)
                state = tk.Label(
                    row,
                    text="待",
                    bg=COLORS["surface_alt"],
                    fg=COLORS["muted_light"],
                    font=("Microsoft YaHei UI", 8),
                    width=3,
                )
                state.pack(side="right", padx=(2, 5))
                self.rows[agent_id] = (dot, state)

    def reset(self, enabled_analysts: tuple[str, ...]) -> None:
        enabled = set(enabled_analysts)
        for agent_id, (dot, state) in self.rows.items():
            disabled = agent_id in {"market", "sentiment", "news", "fundamentals"} and agent_id not in enabled
            dot.configure(fg=COLORS["border"] if disabled else COLORS["muted_light"])
            state.configure(
                text="关" if disabled else "待",
                fg=COLORS["muted_light"],
            )

    def update_status(self, agent_id: str, status: str) -> None:
        row = self.rows.get(agent_id)
        if row is None:
            return
        dot, state = row
        if status == "running":
            dot.configure(fg=COLORS["blue"])
            state.configure(text="运行", fg=COLORS["blue"])
        elif status == "completed":
            dot.configure(fg=COLORS["green"])
            state.configure(text="完成", fg=COLORS["green"])
        else:
            dot.configure(fg=COLORS["red"])
            state.configure(text="异常", fg=COLORS["red"])


@dataclass(frozen=True)
class MarketRequest:
    source: str
    symbol: str
    start: str
    end: str
    adjust: str
    auto_adjust: bool
    csv_path: str
    seed: int


@dataclass(frozen=True)
class ResearchRequest:
    source: str
    symbol: str
    start: str
    end: str
    adjust: str
    auto_adjust: bool
    csv_path: str
    seed: int
    fetch_details: bool
    mode: str
    selected_analysts: tuple[str, ...]
    debate_rounds: int
    risk_rounds: int
    fallback_to_offline: bool
    output_dir: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_temperature: float
    llm_timeout: int


class RavenWatchAgentsDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Raven Watch Agents 中文投研工作台")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg=COLORS["bg"])

        today = datetime.now().date()
        self.source = StringVar(value="demo")
        self.symbol = StringVar(value="ALPHA")
        self.stock_choice = StringVar(value=default_stock_choice("demo"))
        self.analysis_date = StringVar(value=today.isoformat())
        self.history_days = IntVar(value=420)
        self.kline_period = StringVar(value="120")
        self.adjust = StringVar(value="qfq")
        self.auto_adjust = BooleanVar(value=True)
        self.csv_path = StringVar(value="")
        self.seed = IntVar(value=7)
        self.fetch_details = BooleanVar(value=True)

        self.agent_mode = StringVar(value="离线规则")
        self.analyst_market = BooleanVar(value=True)
        self.analyst_sentiment = BooleanVar(value=True)
        self.analyst_news = BooleanVar(value=True)
        self.analyst_fundamentals = BooleanVar(value=True)
        self.debate_rounds = IntVar(value=1)
        self.risk_rounds = IntVar(value=1)
        self.fallback_to_offline = BooleanVar(value=True)

        self.provider_preset = StringVar(value="OpenAI")
        self.llm_base_url = StringVar(value="https://api.openai.com/v1")
        self.llm_model = StringVar(value="")
        self.llm_api_key = StringVar(value="")
        self.llm_temperature = DoubleVar(value=0.2)
        self.llm_timeout = IntVar(value=120)
        self.output_dir = StringVar(value=str(DEFAULT_OUTPUT))

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.worker_kind: str | None = None
        self.last_result: ResearchResult | None = None
        self.current_context: ResearchContext | None = None
        self.last_output_dir: Path | None = None
        self.report_title_to_id: dict[str, str] = {}

        self.run_button: ttk.Button | None = None
        self.kline_button: ttk.Button | None = None
        self.cancel_button: ttk.Button | None = None
        self.open_button: ttk.Button | None = None
        self.progress_bar: ttk.Progressbar | None = None
        self.progress_label: tk.Label | None = None
        self.agent_pipeline: AgentPipelineView | None = None
        self.summary_text: ScrolledText | None = None
        self.report_selector: ttk.Combobox | None = None
        self.report_text: ScrolledText | None = None
        self.stock_selector: ttk.Combobox | None = None
        self.research_notebook: ttk.Notebook | None = None
        self.kline_page: ttk.Frame | None = None
        self.kline_container: tk.Frame | None = None
        self.kline_placeholder: tk.Label | None = None
        self.kline_canvas: FigureCanvasTkAgg | None = None
        self.kline_figure = None
        self.chart_label: tk.Label | None = None
        self.chart_photo: ImageTk.PhotoImage | None = None
        self.decision_action: tk.Label | None = None
        self.decision_confidence: tk.Label | None = None
        self.decision_allocation: tk.Label | None = None
        self.decision_risk: tk.Label | None = None
        self.decision_reason: tk.Label | None = None
        self.quality_text: Text | None = None
        self.app_status_label: tk.Label | None = None
        self.market_context_label: tk.Label | None = None
        self.page_frames: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.backtest_app: QuantStarterApp | None = None
        self.active_page_id = "research"

        self._configure_styles()
        self._build_ui()
        self._refresh_stock_choices(reset=True)
        self.symbol.trace_add("write", lambda *_args: self._refresh_market_context())
        self.root.bind("<Control-Return>", lambda _event: self._start_research())
        self.root.bind("<Control-k>", self._run_kline_shortcut)
        self.root.bind("<Escape>", lambda _event: self._cancel_research())
        self.root.bind("<Alt-Key-1>", lambda _event: self._show_page("research"))
        self.root.bind("<Alt-Key-2>", lambda _event: self._show_page("backtest"))
        self.root.bind("<Alt-Key-3>", lambda _event: self._show_page("settings"))
        self.root.bind("<Control-b>", self._run_backtest_shortcut)
        self.root.after(120, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.option_add("*Font", ("Microsoft YaHei UI", 9))
        self.root.option_add("*tearOff", False)
        style.configure(
            ".",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Content.TFrame", background=COLORS["surface"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure(
            "TButton",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["surface"],
            darkcolor=COLORS["surface"],
            padding=(12, 7),
        )
        style.map(
            "TButton",
            background=[("active", COLORS["surface_alt"]), ("disabled", COLORS["bg"])],
            foreground=[("disabled", COLORS["muted_light"])],
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["teal"],
            foreground="#FFFFFF",
            bordercolor=COLORS["teal"],
            lightcolor=COLORS["teal"],
            darkcolor=COLORS["teal"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(18, 8),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#066E71"), ("disabled", "#9EBFC0")],
            foreground=[("disabled", "#EEF5F5")],
        )
        style.configure(
            "Quiet.TButton",
            background=COLORS["surface_alt"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            padding=(10, 6),
        )
        style.configure(
            "TEntry",
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=(7, 6),
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["surface"],
            background=COLORS["surface"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=(7, 5),
            arrowsize=14,
        )
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["surface"])])
        style.configure(
            "TSpinbox",
            fieldbackground=COLORS["surface"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=(7, 5),
        )
        style.configure("TCheckbutton", background=COLORS["bg"])
        style.configure("Panel.TCheckbutton", background=COLORS["surface"])
        style.map(
            "Panel.TCheckbutton",
            background=[("active", COLORS["surface"])],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["teal"],
            troughcolor=COLORS["border"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["teal"],
            darkcolor=COLORS["teal"],
            thickness=6,
        )
        style.configure(
            "TNotebook",
            background=COLORS["surface"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            borderwidth=0,
            padding=(14, 8),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["surface_alt"])],
            foreground=[("selected", COLORS["teal"])],
        )

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = tk.Frame(self.root, bg=COLORS["sidebar"], width=208)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)

        brand = tk.Frame(sidebar, bg=COLORS["sidebar"])
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 24))
        mark = tk.Frame(brand, bg=COLORS["teal"], width=34, height=34)
        mark.pack(side="left")
        mark.pack_propagate(False)
        tk.Label(
            mark,
            text="TA",
            bg=COLORS["teal"],
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
        ).pack(expand=True)
        brand_text = tk.Frame(brand, bg=COLORS["sidebar"])
        brand_text.pack(side="left", padx=(10, 0))
        tk.Label(
            brand_text,
            text="Raven Watch Agents",
            bg=COLORS["sidebar"],
            fg="#FFFFFF",
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            brand_text,
            text="中文量化工作台",
            bg=COLORS["sidebar"],
            fg="#AAB4BE",
            font=("Microsoft YaHei UI", 8),
            anchor="w",
        ).pack(anchor="w", pady=(1, 0))

        nav = tk.Frame(sidebar, bg=COLORS["sidebar"])
        nav.grid(row=1, column=0, sticky="ew")
        self._add_nav_button(nav, "research", "智能体投研")
        self._add_nav_button(nav, "backtest", "策略回测")
        self._add_nav_button(nav, "settings", "模型与数据")

        sidebar_footer = tk.Frame(sidebar, bg=COLORS["sidebar"])
        sidebar_footer.grid(row=3, column=0, sticky="sew", padx=18, pady=18)
        tk.Frame(sidebar_footer, bg="#3A424B", height=1).pack(fill="x", pady=(0, 14))
        tk.Label(
            sidebar_footer,
            text="研究学习工具\n不自动下单",
            bg=COLORS["sidebar"],
            fg="#8F9BA6",
            font=("Microsoft YaHei UI", 8),
            justify="left",
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            sidebar_footer,
            text="v0.5.1 desktop",
            bg=COLORS["sidebar"],
            fg="#66717C",
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        shell = tk.Frame(self.root, bg=COLORS["bg"])
        shell.grid(row=0, column=1, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        page_container = tk.Frame(shell, bg=COLORS["bg"])
        page_container.grid(row=0, column=0, sticky="nsew")
        page_container.columnconfigure(0, weight=1)
        page_container.rowconfigure(0, weight=1)

        for page_id in ("research", "backtest", "settings"):
            page = tk.Frame(page_container, bg=COLORS["bg"])
            page.grid(row=0, column=0, sticky="nsew")
            self.page_frames[page_id] = page

        status_bar = tk.Frame(
            shell,
            bg=COLORS["surface"],
            height=28,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        status_bar.grid(row=1, column=0, sticky="ew")
        status_bar.grid_propagate(False)
        self.app_status_label = tk.Label(
            status_bar,
            text="系统就绪",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            anchor="w",
        )
        self.app_status_label.pack(side="left", padx=14, fill="y")
        tk.Label(
            status_bar,
            text="历史表现不代表未来收益",
            bg=COLORS["surface"],
            fg=COLORS["muted_light"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side="right", padx=14, fill="y")

        self._build_research_page(self.page_frames["research"])
        self._build_backtest_page(self.page_frames["backtest"])
        self._build_settings_page(self.page_frames["settings"])
        self._show_page("research")
        self._refresh_market_context()

    def _add_nav_button(self, parent: tk.Frame, page_id: str, text: str) -> None:
        button = tk.Button(
            parent,
            text=text,
            command=lambda target=page_id: self._show_page(target),
            bg=COLORS["sidebar"],
            fg="#C3CBD3",
            activebackground=COLORS["sidebar_hover"],
            activeforeground="#FFFFFF",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            anchor="w",
            padx=22,
            pady=11,
            cursor="hand2",
            font=("Microsoft YaHei UI", 10),
        )
        button.pack(fill="x", pady=1)
        self.nav_buttons[page_id] = button

    def _show_page(self, page_id: str) -> None:
        page = self.page_frames.get(page_id)
        if page is None:
            return
        self.active_page_id = page_id
        page.tkraise()
        for key, button in self.nav_buttons.items():
            active = key == page_id
            button.configure(
                bg=COLORS["sidebar_active"] if active else COLORS["sidebar"],
                fg="#FFFFFF" if active else "#C3CBD3",
                font=("Microsoft YaHei UI", 10, "bold" if active else "normal"),
            )
        if self.app_status_label is not None:
            if page_id == "backtest":
                status = "策略回测工作台就绪"
            elif page_id == "settings":
                status = "模型与数据配置"
            elif self.current_context is not None:
                status = f"{self.current_context.symbol} 行情已载入"
            else:
                status = "系统就绪"
            self.app_status_label.configure(text=status)

    def _run_kline_shortcut(self, _event=None) -> None:
        if self.active_page_id == "backtest" and self.backtest_app is not None:
            self.backtest_app._start_kline_preview()
            return
        if self.active_page_id != "research":
            self._show_page("research")
        self._start_kline_preview()

    def _run_backtest_shortcut(self, _event=None) -> None:
        self._show_page("backtest")
        if self.backtest_app is not None:
            self.backtest_app._run_async()

    @staticmethod
    def _panel(parent: tk.Misc) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )

    @staticmethod
    def _caption(parent: tk.Misc, text: str, *, column: int = 0) -> tk.Label:
        label = tk.Label(
            parent,
            text=text,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            anchor="w",
        )
        label.grid(row=0, column=column, sticky="w", pady=(0, 4))
        return label

    @staticmethod
    def _make_readonly_text(parent: tk.Misc, initial: str = "") -> ScrolledText:
        widget = ScrolledText(
            parent,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            borderwidth=0,
            relief="flat",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["teal_soft"],
            padx=14,
            pady=12,
        )
        widget.insert(END, initial)
        widget.configure(state="disabled")
        return widget

    def _build_research_page(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        title_block = tk.Frame(header, bg=COLORS["surface"])
        title_block.grid(row=0, column=0, sticky="w", padx=20, pady=14)
        tk.Label(
            title_block,
            text="智能体投研工作台",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 17, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="分析、辩论、交易、风控与组合决策",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))
        self.market_context_label = tk.Label(
            header,
            text="",
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
            padx=12,
            pady=7,
        )
        self.market_context_label.grid(row=0, column=1, sticky="e", padx=(0, 10))
        self.kline_button = ttk.Button(
            header,
            text="查看K线",
            command=self._start_kline_preview,
            style="Quiet.TButton",
        )
        self.kline_button.grid(row=0, column=2, padx=(0, 8))
        self.cancel_button = ttk.Button(
            header,
            text="取消",
            command=self._cancel_research,
            state="disabled",
            style="Quiet.TButton",
        )
        self.cancel_button.grid(row=0, column=3, padx=(0, 8))
        self.run_button = ttk.Button(
            header,
            text="开始投研",
            command=self._start_research,
            style="Primary.TButton",
        )
        self.run_button.grid(row=0, column=4, padx=(0, 20))

        body = tk.Frame(parent, bg=COLORS["bg"])
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = self._panel(body)
        controls.configure(width=286)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.grid_propagate(False)
        self._build_research_controls(controls)

        center = tk.Frame(body, bg=COLORS["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=12)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(1, weight=1)
        self._build_pipeline_panel(center)
        self._build_report_panel(center)

        right = tk.Frame(body, bg=COLORS["bg"], width=272)
        right.grid(row=0, column=2, sticky="nse")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._build_decision_panel(right)
        self._build_quality_panel(right)

    def _build_research_controls(self, parent: tk.Frame) -> None:
        content = tk.Frame(parent, bg=COLORS["surface"])
        content.pack(fill="both", expand=True, padx=14, pady=14)
        content.columnconfigure((0, 1), weight=1)

        tk.Label(
            content,
            text="研究配置",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self._form_label(content, "数据来源", 1, 0, 2)
        source_box = ttk.Combobox(
            content,
            textvariable=self.source,
            values=("demo", "a-share", "nasdaq", "csv"),
            state="readonly",
        )
        source_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        source_box.bind("<<ComboboxSelected>>", self._on_source_changed)

        self._form_label(content, "选择个股 / 输入代码", 3, 0, 2)
        self.stock_selector = ttk.Combobox(
            content,
            textvariable=self.stock_choice,
            values=stock_choice_labels(self.source.get()),
            state="normal",
        )
        self.stock_selector.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        self.stock_selector.bind("<<ComboboxSelected>>", self._on_stock_selected)
        self.stock_selector.bind("<Return>", self._on_stock_entered)

        self._form_label(content, "分析日期", 5, 0)
        self._form_label(content, "历史天数", 5, 1)
        ttk.Entry(content, textvariable=self.analysis_date).grid(
            row=6, column=0, sticky="ew", padx=(0, 4), pady=(0, 8)
        )
        ttk.Spinbox(
            content,
            textvariable=self.history_days,
            from_=120,
            to=2000,
            increment=30,
        ).grid(row=6, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))

        self._form_label(content, "A 股复权", 7, 0)
        self._form_label(content, "模拟种子", 7, 1)
        ttk.Combobox(
            content,
            textvariable=self.adjust,
            values=("qfq", "hfq", "none"),
            state="readonly",
        ).grid(row=8, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        ttk.Spinbox(content, textvariable=self.seed, from_=1, to=9999).grid(
            row=8, column=1, sticky="ew", padx=(4, 0), pady=(0, 8)
        )

        self._form_label(content, "本地 CSV", 9, 0, 2)
        csv_row = tk.Frame(content, bg=COLORS["surface"])
        csv_row.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        csv_row.columnconfigure(0, weight=1)
        ttk.Entry(csv_row, textvariable=self.csv_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            csv_row, text="浏览", command=self._choose_csv, style="Quiet.TButton"
        ).grid(row=0, column=1, padx=(6, 0))

        ttk.Checkbutton(
            content,
            text="获取基本面与新闻",
            variable=self.fetch_details,
            style="Panel.TCheckbutton",
        ).grid(row=11, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            content,
            text="纳斯达克自动复权",
            variable=self.auto_adjust,
            style="Panel.TCheckbutton",
        ).grid(row=12, column=0, columnspan=2, sticky="w")

        tk.Frame(content, bg=COLORS["border"], height=1).grid(
            row=13, column=0, columnspan=2, sticky="ew", pady=13
        )
        tk.Label(
            content,
            text="智能体编排",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._form_label(content, "运行模式", 15, 0, 2)
        mode_box = ttk.Combobox(
            content,
            textvariable=self.agent_mode,
            values=("离线规则", "在线 LLM"),
            state="readonly",
        )
        mode_box.grid(row=16, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)

        analysts = tk.Frame(content, bg=COLORS["surface"])
        analysts.grid(row=17, column=0, columnspan=2, sticky="ew")
        analysts.columnconfigure((0, 1), weight=1)
        for row, column, text, variable in (
            (0, 0, "技术", self.analyst_market),
            (0, 1, "情绪", self.analyst_sentiment),
            (1, 0, "新闻", self.analyst_news),
            (1, 1, "基本面", self.analyst_fundamentals),
        ):
            ttk.Checkbutton(
                analysts,
                text=text,
                variable=variable,
                style="Panel.TCheckbutton",
            ).grid(row=row, column=column, sticky="w")

        self._form_label(content, "多空轮数", 18, 0)
        self._form_label(content, "风控轮数", 18, 1)
        ttk.Spinbox(
            content, textvariable=self.debate_rounds, from_=1, to=3
        ).grid(row=19, column=0, sticky="ew", padx=(0, 4))
        ttk.Spinbox(content, textvariable=self.risk_rounds, from_=1, to=3).grid(
            row=19, column=1, sticky="ew", padx=(4, 0)
        )

    @staticmethod
    def _form_label(
        parent: tk.Misc,
        text: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> None:
        tk.Label(
            parent,
            text=text,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            anchor="w",
        ).grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="w",
            pady=(0, 3),
        )

    def _build_pipeline_panel(self, parent: tk.Frame) -> None:
        panel = self._panel(parent)
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        header = tk.Frame(panel, bg=COLORS["surface"])
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="协作流水线",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.progress_label = tk.Label(
            header,
            text="等待开始",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        )
        self.progress_label.grid(row=0, column=1, sticky="e")
        self.progress_bar = ttk.Progressbar(panel, mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=14)
        self.agent_pipeline = AgentPipelineView(panel)
        self.agent_pipeline.grid(row=2, column=0, sticky="ew", padx=14, pady=(10, 13))
        self.agent_pipeline.reset(self._selected_analysts())

    def _build_report_panel(self, parent: tk.Frame) -> None:
        panel = self._panel(parent)
        panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        tk.Label(
            panel,
            text="研究证据",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(11, 3))
        notebook = ttk.Notebook(panel)
        self.research_notebook = notebook
        notebook.grid(row=1, column=0, sticky="nsew", padx=1, pady=(0, 1))
        summary_page = ttk.Frame(notebook, style="Content.TFrame")
        report_page = ttk.Frame(notebook, style="Content.TFrame")
        kline_page = ttk.Frame(notebook, style="Content.TFrame")
        chart_page = ttk.Frame(notebook, style="Content.TFrame")
        self.kline_page = kline_page
        notebook.add(summary_page, text="决策摘要")
        notebook.add(report_page, text="代理报告")
        notebook.add(kline_page, text="K线图")
        notebook.add(chart_page, text="综合图")

        summary_page.columnconfigure(0, weight=1)
        summary_page.rowconfigure(0, weight=1)
        self.summary_text = self._make_readonly_text(
            summary_page,
            "选择标的并开始投研后，此处汇总最终决策、关键证据与风险边界。",
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        report_page.columnconfigure(0, weight=1)
        report_page.rowconfigure(1, weight=1)
        self.report_selector = ttk.Combobox(report_page, state="readonly")
        self.report_selector.grid(row=0, column=0, sticky="ew", padx=12, pady=(9, 5))
        self.report_selector.bind("<<ComboboxSelected>>", self._show_selected_report)
        self.report_text = self._make_readonly_text(report_page)
        self.report_text.grid(row=1, column=0, sticky="nsew")

        kline_page.columnconfigure(0, weight=1)
        kline_page.rowconfigure(1, weight=1)
        kline_toolbar = tk.Frame(kline_page, bg=COLORS["surface"])
        kline_toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(7, 4))
        kline_toolbar.columnconfigure(0, weight=1)
        period_box = ttk.Combobox(
            kline_toolbar,
            textvariable=self.kline_period,
            values=("60", "120", "250", "500", "全部"),
            state="readonly",
            width=8,
        )
        period_box.grid(row=0, column=1, padx=(0, 6))
        period_box.bind("<<ComboboxSelected>>", self._refresh_kline_chart)
        ttk.Button(
            kline_toolbar,
            text="刷新",
            command=self._refresh_kline_chart,
            style="Quiet.TButton",
        ).grid(row=0, column=2)
        self.kline_container = tk.Frame(kline_page, bg=COLORS["surface"])
        self.kline_container.grid(row=1, column=0, sticky="nsew")
        self.kline_container.columnconfigure(0, weight=1)
        self.kline_container.rowconfigure(0, weight=1)
        self.kline_placeholder = tk.Label(
            self.kline_container,
            text="暂无K线",
            bg=COLORS["surface"],
            fg=COLORS["muted_light"],
            font=("Microsoft YaHei UI", 9),
        )
        self.kline_placeholder.grid(row=0, column=0, sticky="nsew")

        chart_page.columnconfigure(0, weight=1)
        chart_page.rowconfigure(0, weight=1)
        self.chart_label = tk.Label(
            chart_page,
            text="投研完成后显示行情、均线与信号证据",
            bg=COLORS["surface"],
            fg=COLORS["muted_light"],
            font=("Microsoft YaHei UI", 9),
            anchor="center",
        )
        self.chart_label.grid(row=0, column=0, sticky="nsew")

    def _build_decision_panel(self, parent: tk.Frame) -> None:
        panel = self._panel(parent)
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        content = tk.Frame(panel, bg=COLORS["surface"])
        content.grid(row=0, column=0, sticky="ew", padx=15, pady=14)
        content.columnconfigure((0, 1), weight=1)
        tk.Label(
            content,
            text="组合经理决策",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.decision_action = tk.Label(
            content,
            text="等待研究",
            bg=COLORS["surface"],
            fg=COLORS["muted_light"],
            font=("Microsoft YaHei UI", 25, "bold"),
            anchor="w",
        )
        self.decision_action.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 12))
        tk.Frame(content, bg=COLORS["border"], height=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 11)
        )
        self._metric_label(content, "置信度", 3, 0)
        self._metric_label(content, "目标仓位", 3, 1)
        self.decision_confidence = self._metric_value(content, "--", 4, 0)
        self.decision_allocation = self._metric_value(content, "--", 4, 1)
        self._metric_label(content, "止损 / 止盈", 5, 0, 2)
        self.decision_risk = self._metric_value(content, "--", 6, 0, 2)
        self.decision_reason = tk.Label(
            content,
            text="完成研究后显示组合层理由。",
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            justify="left",
            anchor="nw",
            wraplength=224,
            padx=9,
            pady=8,
        )
        self.decision_reason.grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(12, 10)
        )
        self.open_button = ttk.Button(
            content,
            text="打开报告目录",
            command=self._open_output,
            state="disabled",
            style="Quiet.TButton",
        )
        self.open_button.grid(row=8, column=0, columnspan=2, sticky="ew")

    @staticmethod
    def _metric_label(
        parent: tk.Misc,
        text: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> None:
        tk.Label(
            parent,
            text=text,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            anchor="w",
        ).grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=(4, 1))

    @staticmethod
    def _metric_value(
        parent: tk.Misc,
        text: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> tk.Label:
        label = tk.Label(
            parent,
            text=text,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        )
        label.grid(row=row, column=column, columnspan=columnspan, sticky="w")
        return label

    def _build_quality_panel(self, parent: tk.Frame) -> None:
        panel = self._panel(parent)
        panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        tk.Label(
            panel,
            text="数据与运行状态",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        self.quality_text = Text(
            panel,
            wrap="word",
            borderwidth=0,
            relief="flat",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
            padx=14,
            pady=6,
            height=8,
        )
        self.quality_text.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self.quality_text.insert(END, "等待数据检查。")
        self.quality_text.configure(state="disabled")

    def _build_backtest_page(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)
        self.backtest_app = QuantStarterApp(parent, embedded=True)

    def _build_settings_page(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="模型与数据设置",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(anchor="w", padx=20, pady=(14, 2))
        tk.Label(
            header,
            text="OpenAI 兼容模型、报告目录与故障回退",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=20, pady=(0, 14))

        body = tk.Frame(parent, bg=COLORS["bg"])
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        body.columnconfigure((0, 1), weight=1, uniform="settings")
        body.rowconfigure(0, weight=1)
        model_panel = self._panel(body)
        model_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        output_panel = self._panel(body)
        output_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_model_form(model_panel)
        self._build_output_form(output_panel)

    def _build_model_form(self, parent: tk.Frame) -> None:
        form = tk.Frame(parent, bg=COLORS["surface"])
        form.pack(fill="both", expand=True, padx=20, pady=18)
        form.columnconfigure((0, 1), weight=1)
        tk.Label(
            form,
            text="在线模型",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))
        self._form_label(form, "接口预设", 1, 0, 2)
        preset = ttk.Combobox(
            form,
            textvariable=self.provider_preset,
            values=("OpenAI", "DeepSeek", "Qwen", "Ollama", "自定义"),
            state="readonly",
        )
        preset.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        preset.bind("<<ComboboxSelected>>", self._apply_provider_preset)
        self._form_label(form, "Base URL", 3, 0, 2)
        ttk.Entry(form, textvariable=self.llm_base_url).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 12)
        )
        self._form_label(form, "模型 ID", 5, 0, 2)
        ttk.Entry(form, textvariable=self.llm_model).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12)
        )
        self._form_label(form, "API Key", 7, 0, 2)
        ttk.Entry(form, textvariable=self.llm_api_key, show="●").grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(0, 12)
        )
        self._form_label(form, "Temperature", 9, 0)
        self._form_label(form, "超时（秒）", 9, 1)
        ttk.Entry(form, textvariable=self.llm_temperature).grid(
            row=10, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Entry(form, textvariable=self.llm_timeout).grid(
            row=10, column=1, sticky="ew", padx=(5, 0)
        )
        ttk.Checkbutton(
            form,
            text="调用失败时回退到离线规则",
            variable=self.fallback_to_offline,
            style="Panel.TCheckbutton",
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _build_output_form(self, parent: tk.Frame) -> None:
        form = tk.Frame(parent, bg=COLORS["surface"])
        form.pack(fill="both", expand=True, padx=20, pady=18)
        form.columnconfigure(0, weight=1)
        tk.Label(
            form,
            text="本地输出",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 16))
        self._form_label(form, "研究报告目录", 1, 0)
        output_row = tk.Frame(form, bg=COLORS["surface"])
        output_row.grid(row=2, column=0, sticky="ew")
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_dir).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(
            output_row,
            text="选择目录",
            command=self._choose_output,
            style="Quiet.TButton",
        ).grid(row=0, column=1, padx=(8, 0))
        tk.Frame(form, bg=COLORS["border"], height=1).grid(
            row=3, column=0, sticky="ew", pady=22
        )
        tk.Label(
            form,
            text="凭据处理",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=4, column=0, sticky="w")
        tk.Label(
            form,
            text=(
                "API Key 仅保存在当前进程内，不写入报告、记忆日志或配置文件。"
                "在线模式会向所选服务商发送行情摘要与代理报告。"
            ),
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
            justify="left",
            anchor="nw",
            wraplength=450,
            padx=12,
            pady=12,
        ).grid(row=5, column=0, sticky="ew", pady=(10, 0))

    def _refresh_market_context(self) -> None:
        if self.market_context_label is not None:
            self.market_context_label.configure(
                text=f"{self.source.get()}  ·  {self.symbol.get().strip() or '--'}"
            )

    def _refresh_stock_choices(self, *, reset: bool) -> None:
        source = self.source.get()
        labels = stock_choice_labels(source)
        if self.stock_selector is not None:
            self.stock_selector.configure(values=labels)
        if reset:
            choice = default_stock_choice(source)
            self.stock_choice.set(choice)
            try:
                self.symbol.set(resolve_stock_choice(choice, source))
            except ValueError:
                self.symbol.set("CSV")
        self._refresh_market_context()

    def _on_source_changed(self, _event=None) -> None:
        self._refresh_stock_choices(reset=True)

    def _on_stock_selected(self, _event=None) -> None:
        self._commit_stock_choice(show_error=False)

    def _on_stock_entered(self, _event=None) -> None:
        self._commit_stock_choice(show_error=True)

    def _commit_stock_choice(self, *, show_error: bool) -> bool:
        try:
            symbol = resolve_stock_choice(self.stock_choice.get(), self.source.get())
        except ValueError as exc:
            if show_error:
                messagebox.showerror("个股代码无效", str(exc))
            return False
        self.symbol.set(symbol)
        self.stock_choice.set(display_for_symbol(self.source.get(), symbol))
        self._refresh_market_context()
        return True

    def _on_mode_changed(self, _event=None) -> None:
        if self.agent_mode.get() == "在线 LLM" and not self.llm_model.get().strip():
            self._show_page("settings")

    def _apply_provider_preset(self, _event=None) -> None:
        urls = {
            "OpenAI": "https://api.openai.com/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
            "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "Ollama": "http://localhost:11434/v1",
        }
        selected = self.provider_preset.get()
        if selected in urls:
            self.llm_base_url.set(urls[selected])

    def _choose_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 OHLCV CSV",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if path:
            self.csv_path.set(path)
            self.source.set("csv")
            self._refresh_stock_choices(reset=True)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择报告目录")
        if path:
            self.output_dir.set(path)

    def _selected_analysts(self) -> tuple[str, ...]:
        pairs = (
            ("market", self.analyst_market.get()),
            ("sentiment", self.analyst_sentiment.get()),
            ("news", self.analyst_news.get()),
            ("fundamentals", self.analyst_fundamentals.get()),
        )
        return tuple(name for name, enabled in pairs if enabled)

    def _snapshot_market_request(self) -> MarketRequest:
        symbol = resolve_stock_choice(self.stock_choice.get(), self.source.get())
        self.symbol.set(symbol)
        self.stock_choice.set(display_for_symbol(self.source.get(), symbol))
        end_date = datetime.strptime(self.analysis_date.get().strip(), "%Y-%m-%d").date()
        history_days = int(self.history_days.get())
        if history_days < 120:
            raise ValueError("历史天数至少需要 120 天。")
        start_date = end_date - timedelta(days=history_days)
        if self.source.get() == "csv" and not self.csv_path.get().strip():
            raise ValueError("CSV 数据源必须选择 OHLCV 文件。")
        return MarketRequest(
            source=self.source.get(),
            symbol=symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            adjust=self.adjust.get(),
            auto_adjust=self.auto_adjust.get(),
            csv_path=self.csv_path.get().strip(),
            seed=int(self.seed.get()),
        )

    def _snapshot_request(self) -> ResearchRequest:
        market = self._snapshot_market_request()
        mode = "online" if self.agent_mode.get() == "在线 LLM" else "offline"
        selected = self._selected_analysts()
        if not selected:
            raise ValueError("至少选择一名分析师。")

        return ResearchRequest(
            source=market.source,
            symbol=market.symbol,
            start=market.start,
            end=market.end,
            adjust=market.adjust,
            auto_adjust=market.auto_adjust,
            csv_path=market.csv_path,
            seed=market.seed,
            fetch_details=self.fetch_details.get(),
            mode=mode,
            selected_analysts=selected,
            debate_rounds=int(self.debate_rounds.get()),
            risk_rounds=int(self.risk_rounds.get()),
            fallback_to_offline=self.fallback_to_offline.get(),
            output_dir=self.output_dir.get().strip(),
            llm_base_url=self.llm_base_url.get().strip(),
            llm_model=self.llm_model.get().strip(),
            llm_api_key=self.llm_api_key.get(),
            llm_temperature=float(self.llm_temperature.get()),
            llm_timeout=int(self.llm_timeout.get()),
        )

    def _start_kline_preview(self) -> None:
        if self.worker_kind is not None or (
            self.worker is not None and self.worker.is_alive()
        ):
            return
        try:
            market = self._snapshot_market_request()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("无法加载K线", str(exc))
            return

        self._set_running(True, cancellable=False)
        self.worker_kind = "kline"
        self.current_context = None
        self._clear_kline("正在加载K线...")
        if self.app_status_label is not None:
            self.app_status_label.configure(text=f"正在加载 {market.symbol} K线")
        self.worker = threading.Thread(
            target=self._kline_worker, args=(market,), daemon=True
        )
        self.worker.start()

    def _kline_worker(self, market: MarketRequest) -> None:
        try:
            context = build_research_context(
                source=market.source,
                symbol=market.symbol,
                start=market.start,
                end=market.end,
                adjust=market.adjust,
                auto_adjust=market.auto_adjust,
                csv_path=market.csv_path or None,
                seed=market.seed,
                fetch_details=False,
            )
            self.events.put(("kline", context))
        except Exception as exc:
            self.events.put(("kline_error", exc))

    def _start_research(self) -> None:
        if self.worker_kind is not None or (
            self.worker is not None and self.worker.is_alive()
        ):
            return
        try:
            request_data = self._snapshot_request()
            if request_data.mode == "online":
                LLMSettings(
                    base_url=request_data.llm_base_url,
                    model=request_data.llm_model,
                    api_key=request_data.llm_api_key,
                    temperature=request_data.llm_temperature,
                    timeout_seconds=request_data.llm_timeout,
                ).validate()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("无法开始投研", str(exc))
            return

        self._set_running(True)
        self.worker_kind = "research"
        self.cancel_event.clear()
        self.last_result = None
        self.last_output_dir = None
        self.report_title_to_id.clear()
        self._clear_results()
        self.worker = threading.Thread(
            target=self._research_worker, args=(request_data,), daemon=True
        )
        if self.app_status_label is not None:
            self.app_status_label.configure(
                text=f"正在研究 {request_data.symbol} · {request_data.mode}"
            )
        self.worker.start()

    def _research_worker(self, request_data: ResearchRequest) -> None:
        try:
            self.events.put(("stage", "正在获取行情、基本面与新闻"))
            context = build_research_context(
                source=request_data.source,
                symbol=request_data.symbol,
                start=request_data.start,
                end=request_data.end,
                adjust=request_data.adjust,
                auto_adjust=request_data.auto_adjust,
                csv_path=request_data.csv_path or None,
                seed=request_data.seed,
                fetch_details=request_data.fetch_details,
            )
            if self.cancel_event.is_set():
                raise WorkflowCancelled("用户已取消本次投研。")

            client = None
            if request_data.mode == "online":
                client = OpenAICompatibleClient(
                    LLMSettings(
                        base_url=request_data.llm_base_url,
                        model=request_data.llm_model,
                        api_key=request_data.llm_api_key,
                        temperature=request_data.llm_temperature,
                        timeout_seconds=request_data.llm_timeout,
                    )
                )
            workflow = RavenWatchAgentsWorkflow(
                WorkflowConfig(
                    mode=request_data.mode,
                    selected_analysts=request_data.selected_analysts,
                    debate_rounds=request_data.debate_rounds,
                    risk_rounds=request_data.risk_rounds,
                    fallback_to_offline=request_data.fallback_to_offline,
                ),
                client,
            )
            current_close = context.technical.get("close")
            memory_context = load_memory_context(
                context.symbol,
                DEFAULT_MEMORY,
                current_close=float(current_close) if current_close is not None else None,
            )
            result = workflow.run(
                context,
                progress=lambda event: self.events.put(("progress", event)),
                memory_context=memory_context,
                cancel_event=self.cancel_event,
            )
            run_dir = save_research_result(result, request_data.output_dir)
            append_decision_memory(result, DEFAULT_MEMORY)
            self.events.put(("complete", (result, run_dir)))
        except WorkflowCancelled as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "stage":
                    if self.progress_label is not None:
                        self.progress_label.configure(text=str(payload))
                    if self.app_status_label is not None:
                        self.app_status_label.configure(text=str(payload))
                elif kind == "progress":
                    self._handle_progress(payload)
                elif kind == "complete":
                    result, run_dir = payload
                    self._handle_complete(result, run_dir)
                elif kind == "kline":
                    self._handle_kline(payload)
                elif kind == "kline_error":
                    self._handle_kline_error(payload)
                elif kind == "cancelled":
                    self._handle_cancelled(str(payload))
                elif kind == "error":
                    self._handle_error(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def _handle_progress(self, event: object) -> None:
        if not isinstance(event, ProgressEvent):
            return
        if self.progress_bar is not None:
            self.progress_bar.configure(maximum=event.total, value=event.step)
        if self.progress_label is not None:
            self.progress_label.configure(text=f"{event.step}/{event.total} {event.agent_name}")
        if self.agent_pipeline is not None:
            self.agent_pipeline.update_status(event.agent_id, event.status)

    def _handle_kline(self, payload: object) -> None:
        if not isinstance(payload, ResearchContext):
            self._handle_kline_error(ValueError("K线行情返回了无效结果。"))
            return
        self.current_context = payload
        try:
            self._render_kline_chart(select_tab=True)
        except Exception as exc:
            self._handle_kline_error(exc)
            return

        bars = payload.bars
        provider = payload.technical.get(
            "data_provider_label",
            payload.technical.get("data_provider", payload.market),
        )
        self._replace_text(
            self.quality_text,
            "\n".join(
                [
                    f"市场：{payload.market}",
                    f"标的：{payload.symbol}",
                    f"行情：{len(bars)} 条日线",
                    f"区间：{bars.index.min().date()} 至 {bars.index.max().date()}",
                    f"数据源：{provider}",
                    "",
                    "K线数据检查通过。",
                ]
            ),
        )
        if self.app_status_label is not None:
            self.app_status_label.configure(
                text=f"{payload.symbol} K线已加载 · {len(bars)} 条日线"
            )
        self._set_running(False)

    def _selected_kline_window(self) -> int | None:
        selected = self.kline_period.get().strip()
        if selected == "全部":
            return None
        try:
            return int(selected)
        except ValueError as exc:
            raise ValueError("K线周期无效。") from exc

    def _refresh_kline_chart(self, _event=None) -> None:
        if self.current_context is None:
            return
        try:
            self._render_kline_chart(select_tab=False)
        except Exception as exc:
            self._handle_kline_error(exc)

    def _render_kline_chart(self, *, select_tab: bool) -> None:
        if self.current_context is None or self.kline_container is None:
            return
        figure = build_kline_figure(
            self.current_context.bars,
            self.current_context.symbol,
            self._selected_kline_window(),
        )
        if self.kline_canvas is not None:
            self.kline_canvas.get_tk_widget().destroy()
            self.kline_canvas = None
        if self.kline_figure is not None:
            self.kline_figure.clear()
        self.kline_figure = figure
        if self.kline_placeholder is not None:
            self.kline_placeholder.grid_remove()
        self.kline_canvas = FigureCanvasTkAgg(figure, master=self.kline_container)
        canvas_widget = self.kline_canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=0, bg=COLORS["surface"])
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.kline_canvas.draw()
        if select_tab and self.research_notebook is not None and self.kline_page is not None:
            self.research_notebook.select(self.kline_page)

    def _clear_kline(self, message: str = "暂无K线") -> None:
        if self.kline_canvas is not None:
            self.kline_canvas.get_tk_widget().destroy()
            self.kline_canvas = None
        if self.kline_figure is not None:
            self.kline_figure.clear()
            self.kline_figure = None
        if self.kline_placeholder is not None:
            self.kline_placeholder.configure(text=message)
            self.kline_placeholder.grid()

    def _handle_kline_error(self, exc: object) -> None:
        message = str(exc)
        self._clear_kline("K线加载失败")
        if self.app_status_label is not None:
            self.app_status_label.configure(text="K线加载失败")
        self._set_running(False)
        messagebox.showerror("K线加载失败", message)

    def _handle_complete(self, result: ResearchResult, run_dir: Path) -> None:
        self.last_result = result
        self.current_context = result.context
        self.last_output_dir = run_dir
        decision = result.decision
        action_color = {
            "BUY": COLORS["green"],
            "HOLD": COLORS["amber"],
            "SELL": COLORS["red"],
        }.get(decision.action, COLORS["text"])
        if self.decision_action is not None:
            self.decision_action.configure(text=decision.action_cn, fg=action_color)
        if self.decision_confidence is not None:
            self.decision_confidence.configure(text=f"{decision.confidence}%")
        if self.decision_allocation is not None:
            self.decision_allocation.configure(text=f"{decision.target_allocation:.0%}")
        if self.decision_risk is not None:
            self.decision_risk.configure(
                text=f"-{decision.stop_loss_pct:.1%} / +{decision.take_profit_pct:.1%}"
            )
        if self.decision_reason is not None:
            self.decision_reason.configure(text=decision.rationale)
        if self.progress_label is not None:
            self.progress_label.configure(text="投研完成")
        if self.progress_bar is not None:
            self.progress_bar.configure(value=self.progress_bar.cget("maximum"))
        if self.app_status_label is not None:
            self.app_status_label.configure(
                text=f"{result.context.symbol} 投研完成 · {len(result.reports)} 份代理报告"
            )

        bars = result.context.bars
        quality_lines = [
            f"市场：{result.context.market}",
            f"行情：{len(bars)} 条日线",
            f"区间：{bars.index.min().date()} 至 {bars.index.max().date()}",
            f"基本面字段：{len(result.context.fundamentals)}",
            f"新闻样本：{len(result.context.news)}",
            f"运行模式：{'在线 LLM' if result.mode == 'online' else '离线规则'}",
        ]
        if result.warnings:
            quality_lines.extend(["", "警告"])
            quality_lines.extend(f"· {warning}" for warning in result.warnings)
        else:
            quality_lines.extend(["", "数据检查通过，未发现运行警告。"])
        self._replace_text(self.quality_text, "\n".join(quality_lines))

        summary_lines = [
            f"{result.context.symbol} · {result.context.analysis_date}",
            "",
            f"最终决策：{decision.action_cn} ({decision.action})",
            f"置信度：{decision.confidence}%",
            f"目标仓位：{decision.target_allocation:.1%}",
            f"止损预算：{decision.stop_loss_pct:.1%}",
            f"止盈目标：{decision.take_profit_pct:.1%}",
            f"观察周期：{decision.time_horizon}",
            "",
            "组合经理理由",
            decision.rationale,
            "",
            f"报告目录：{run_dir}",
        ]
        if result.warnings:
            summary_lines.extend(["", "数据与运行警告"])
            summary_lines.extend(f"- {warning}" for warning in result.warnings)
        self._replace_text(self.summary_text, "\n".join(summary_lines))

        display_titles = []
        for report_id, report in result.reports.items():
            title = result.report_titles.get(report_id, report_id)
            unique_title = title
            suffix = 2
            while unique_title in self.report_title_to_id:
                unique_title = f"{title} {suffix}"
                suffix += 1
            self.report_title_to_id[unique_title] = report_id
            display_titles.append(unique_title)
        if self.report_selector is not None:
            self.report_selector.configure(values=display_titles)
            if display_titles:
                self.report_selector.set(display_titles[-1])
                self._show_selected_report()
        if self.open_button is not None:
            self.open_button.configure(state="normal")
        try:
            self._render_kline_chart(select_tab=True)
        except Exception as exc:
            self._clear_kline(f"K线渲染失败：{exc}")
        self._show_chart(run_dir / "research_overview.png")
        self._set_running(False)

    def _show_chart(self, chart_path: Path) -> None:
        if self.chart_label is None:
            return
        try:
            with Image.open(chart_path) as source:
                image = source.copy()
            image.thumbnail((820, 480), Image.Resampling.LANCZOS)
            self.chart_photo = ImageTk.PhotoImage(image)
            self.chart_label.configure(image=self.chart_photo, text="")
        except Exception as exc:
            self.chart_photo = None
            self.chart_label.configure(image="", text=f"图表加载失败：{exc}")

    def _handle_cancelled(self, message: str) -> None:
        if self.progress_label is not None:
            self.progress_label.configure(text="已取消")
        self._replace_text(self.summary_text, message)
        if self.app_status_label is not None:
            self.app_status_label.configure(text="本次投研已取消")
        self._set_running(False)

    def _handle_error(self, exc: object) -> None:
        message = str(exc)
        if self.progress_label is not None:
            self.progress_label.configure(text="运行失败")
        self._replace_text(self.summary_text, "发生错误\n\n" + message)
        self._replace_text(self.quality_text, "运行失败\n\n" + message)
        if self.app_status_label is not None:
            self.app_status_label.configure(text="投研失败")
        self._set_running(False)
        messagebox.showerror("投研失败", message)

    def _show_selected_report(self, _event=None) -> None:
        if self.last_result is None or self.report_selector is None:
            return
        report_id = self.report_title_to_id.get(self.report_selector.get())
        if report_id:
            self._replace_text(self.report_text, self.last_result.reports[report_id])

    @staticmethod
    def _replace_text(widget: Text | None, content: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.insert(END, content)
        widget.configure(state="disabled")

    def _clear_results(self) -> None:
        self.current_context = None
        if self.agent_pipeline is not None:
            self.agent_pipeline.reset(self._selected_analysts())
        if self.progress_bar is not None:
            self.progress_bar.configure(value=0, maximum=1)
        if self.progress_label is not None:
            self.progress_label.configure(text="准备数据")
        self._replace_text(self.summary_text, "正在准备本次多智能体投研...")
        self._replace_text(self.report_text, "")
        if self.report_selector is not None:
            self.report_selector.set("")
            self.report_selector.configure(values=())
        if self.decision_action is not None:
            self.decision_action.configure(text="分析中", fg=COLORS["blue"])
        for label in (
            self.decision_confidence,
            self.decision_allocation,
            self.decision_risk,
        ):
            if label is not None:
                label.configure(text="--")
        if self.decision_reason is not None:
            self.decision_reason.configure(text="智能体正在形成组合层判断。")
        self._replace_text(self.quality_text, "正在获取并检查行情数据。")
        if self.open_button is not None:
            self.open_button.configure(state="disabled")
        self.chart_photo = None
        if self.chart_label is not None:
            self.chart_label.configure(image="", text="正在生成行情证据图...")
        self._clear_kline("正在加载K线...")

    def _set_running(self, running: bool, *, cancellable: bool = True) -> None:
        if self.run_button is not None:
            self.run_button.configure(state="disabled" if running else "normal")
        if self.kline_button is not None:
            self.kline_button.configure(state="disabled" if running else "normal")
        if self.cancel_button is not None:
            self.cancel_button.configure(
                state="normal" if running and cancellable else "disabled"
            )
        if not running:
            self.worker_kind = None

    def _cancel_research(self) -> None:
        if self.worker_kind != "research":
            return
        self.cancel_event.set()
        if self.progress_label is not None:
            self.progress_label.configure(text="正在取消（当前请求结束后停止）")
        if self.app_status_label is not None:
            self.app_status_label.configure(text="正在取消投研")

    def _open_output(self) -> None:
        if self.last_output_dir is not None:
            os.startfile(self.last_output_dir)

    def _open_backtest_window(self) -> None:
        window = Toplevel(self.root)
        QuantStarterApp(window)


def run_smoke_test() -> None:
    output_root = ROOT / "smoke-test-output"
    context = build_research_context(
        source="demo",
        symbol="ALPHA",
        start="2024-01-01",
        end="2025-12-31",
        fetch_details=True,
    )
    workflow = RavenWatchAgentsWorkflow(
        WorkflowConfig(mode="offline", debate_rounds=1, risk_rounds=1)
    )
    result = workflow.run(context)
    run_dir = save_research_result(result, output_root)
    required = (
        "summary.json",
        "full_report.md",
        "market_data.csv",
        "research_overview.png",
        "kline_chart.png",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("Smoke test output is missing: " + ", ".join(missing))
    if len(result.reports) < 10:
        raise RuntimeError("Smoke test did not run the complete agent team")


def run_market_data_smoke_test() -> None:
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=240)
    cases = (("a-share", "300750"), ("nasdaq", "NVDA"))
    for source, symbol in cases:
        context = build_research_context(
            source=source,
            symbol=symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            fetch_details=False,
        )
        if len(context.bars) < 60:
            raise RuntimeError(f"{source} {symbol} returned too few market rows")
        result = RavenWatchAgentsWorkflow(
            WorkflowConfig(mode="offline", debate_rounds=1, risk_rounds=1)
        ).run(context)
        if len(result.reports) < 10:
            raise RuntimeError(f"{source} {symbol} did not complete agent analysis")


def main() -> None:
    if "--data-smoke-test" in sys.argv:
        run_market_data_smoke_test()
        return
    if "--smoke-test" in sys.argv:
        run_smoke_test()
        return

    root = Tk()
    RavenWatchAgentsDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
