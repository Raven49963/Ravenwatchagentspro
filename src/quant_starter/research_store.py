from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re

from quant_starter.agent_workflow import ResearchResult
from quant_starter.kline import save_kline_chart
from quant_starter.research_charts import save_research_overview


def safe_symbol_component(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("股票代码不能用于创建结果目录。")
    return cleaned[:64]


def save_research_result(result: ResearchResult, output_root: str | Path) -> Path:
    """Persist an auditable report bundle and return its run directory."""

    root = Path(output_root).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{safe_symbol_component(result.context.symbol)}_{stamp}"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=False)

    result.context.bars.to_csv(run_dir / "market_data.csv", encoding="utf-8-sig")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary_dict(), handle, ensure_ascii=False, indent=2, default=str)

    full_report = [
        f"# {result.context.symbol} 多智能体投研报告",
        "",
        f"- 分析日期：{result.context.analysis_date}",
        f"- 数据市场：{result.context.market}",
        f"- 运行模式：{result.mode}",
        f"- 最终决策：{result.decision.action_cn} ({result.decision.action})",
        f"- 置信度：{result.decision.confidence}%",
        f"- 目标仓位：{result.decision.target_allocation:.1%}",
        f"- 止损预算：{result.decision.stop_loss_pct:.1%}",
        f"- 止盈目标：{result.decision.take_profit_pct:.1%}",
        "",
        "> 本报告仅用于量化研究学习，不构成投资建议，也不会自动下单。",
    ]

    for index, (report_id, report) in enumerate(result.reports.items(), start=1):
        title = result.report_titles.get(report_id, report_id)
        filename = f"{index:02d}_{safe_symbol_component(report_id)}.md"
        (report_dir / filename).write_text(report, encoding="utf-8")
        full_report.extend(["", "---", "", f"# {title}", "", report])

    if result.warnings:
        full_report.extend(["", "---", "", "# 数据与运行警告", ""])
        full_report.extend(f"- {warning}" for warning in result.warnings)

    (run_dir / "full_report.md").write_text(
        "\n".join(full_report), encoding="utf-8"
    )
    save_research_overview(result, run_dir)
    save_kline_chart(
        result.context.bars,
        result.context.symbol,
        run_dir / "kline_chart.png",
    )
    return run_dir


def append_decision_memory(result: ResearchResult, memory_file: str | Path) -> None:
    """Append one sanitized decision record; reports and API settings are excluded."""

    path = Path(memory_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at": result.completed_at,
        "symbol": result.context.symbol,
        "analysis_date": result.context.analysis_date,
        "market": result.context.market,
        "decision": asdict(result.decision),
        "close": result.context.technical.get("close"),
        "return_20d_at_decision": result.context.technical.get("return_20d"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_memory_context(
    symbol: str,
    memory_file: str | Path,
    limit: int = 3,
    current_close: float | None = None,
) -> str:
    path = Path(memory_file).expanduser()
    if not path.exists():
        return ""

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("symbol", "")).upper() == symbol.upper():
            records.append(record)

    selected = records[-max(1, limit) :]
    if not selected:
        return ""
    lines = []
    for item in selected:
        decision = item.get("decision", {})
        line = (
            f"{item.get('analysis_date')}: "
            f"{decision.get('action', 'UNKNOWN')}, "
            f"confidence={decision.get('confidence', 'N/A')}, "
            f"close={item.get('close', 'N/A')}"
        )
        try:
            old_close = float(item.get("close"))
            if current_close is not None and old_close > 0:
                change = current_close / old_close - 1.0
                line += f", price_change_to_now={change:+.2%}"
        except (TypeError, ValueError):
            pass
        lines.append(line)
    return "\n".join(lines)
