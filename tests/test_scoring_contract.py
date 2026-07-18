from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from quant_starter.data import DemoMarketConfig, generate_demo_ohlcv
from quant_starter.factors import analyze_composite
from quant_starter.local_evidence import build_local_evidence


class ScoringBehaviorContractTests(unittest.TestCase):
    def test_composite_research_keeps_its_public_result_contract(self) -> None:
        bars = generate_demo_ohlcv(
            "ALPHA",
            DemoMarketConfig(start="2021-01-01", end="2026-01-01", seed=41),
        )

        result = analyze_composite(bars)

        self.assertEqual(
            {
                "score": result.score,
                "directional_score": result.directional_score,
                "raw_directional_score": result.raw_directional_score,
                "signal_strength": result.signal_strength,
                "agreement": result.agreement,
                "factor_stability": result.factor_stability,
                "factor_coverage": result.factor_coverage,
                "calibration_factor": result.calibration_factor,
                "action": result.action,
                "action_label": result.action_label,
                "confidence": result.confidence,
                "target_position": result.target_position,
                "stop_loss_pct": result.stop_loss_pct,
                "take_profit_pct": result.take_profit_pct,
                "risk_level": result.risk_level,
                "summary": result.summary,
            },
            {
                "score": 69.14,
                "directional_score": 38.28,
                "raw_directional_score": 43.79,
                "signal_strength": 38.28,
                "agreement": 0.819786,
                "factor_stability": 0.2945,
                "factor_coverage": 1.0,
                "calibration_factor": 0.874275,
                "action": "BUY",
                "action_label": "偏多",
                "confidence": 75,
                "target_position": 0.2647,
                "stop_loss_pct": 0.0697,
                "take_profit_pct": 0.1394,
                "risk_level": "中",
                "summary": (
                    "当前处于多头趋势，研判评分 69.1/100（多头方向较强，"
                    "方向强度 38.3），一致度 82%，历史分段稳定度 29%；"
                    "建议目标仓位 26%，风险等级中。"
                ),
            },
        )

    def test_local_evidence_keeps_its_public_result_contract(self) -> None:
        result = build_local_evidence(
            technical_score=-80,
            technical_confidence=90,
            quant_validation={
                "available": True,
                "latest_score": -0.8,
                "robustness_score": 90,
                "folds": [{}, {}, {}],
                "verdict": "偏空",
            },
            fundamentals={
                "available": True,
                "directional_score": -75,
                "score": 12.5,
                "confidence": 90,
                "available_metrics": 12,
                "total_metrics": 16,
            },
            news={
                "available": True,
                "directional_score": -75,
                "score": 12.5,
                "confidence": 90,
                "article_count": 8,
                "official_ratio": 0.5,
            },
        )

        self.assertEqual(
            {
                key: result[key]
                for key in (
                    "score",
                    "directional_score",
                    "raw_directional_score",
                    "signal_strength",
                    "label",
                    "confidence",
                    "agreement",
                    "calibration_factor",
                    "coverage",
                    "components",
                    "conflicts",
                    "missing_components",
                    "summary",
                )
            },
            {
                "score": 13.6,
                "directional_score": -72.8,
                "raw_directional_score": -78.25,
                "signal_strength": 72.8,
                "label": "空头风险占优",
                "confidence": 89,
                "agreement": 0.97725,
                "calibration_factor": 0.930342,
                "coverage": 1.0,
                "components": [
                    {
                        "key": "technical",
                        "label": "技术因子",
                        "available": True,
                        "score": 10.0,
                        "directional_score": -80.0,
                        "direction": "明显偏空",
                        "confidence": 90,
                        "base_weight": 0.35,
                        "effective_weight": 0.35,
                        "detail": "27 项价格、波动、流动性与尾部风险因子",
                    },
                    {
                        "key": "quant",
                        "label": "样本外验证",
                        "available": True,
                        "score": 10.0,
                        "directional_score": -80.0,
                        "direction": "明显偏空",
                        "confidence": 90,
                        "base_weight": 0.3,
                        "effective_weight": 0.3,
                        "detail": "3 折 · 偏空",
                    },
                    {
                        "key": "fundamentals",
                        "label": "基本面",
                        "available": True,
                        "score": 12.5,
                        "directional_score": -75.0,
                        "direction": "明显偏空",
                        "confidence": 90,
                        "base_weight": 0.2,
                        "effective_weight": 0.2,
                        "detail": "12 / 16 项",
                    },
                    {
                        "key": "news",
                        "label": "新闻事件",
                        "available": True,
                        "score": 12.5,
                        "directional_score": -75.0,
                        "direction": "明显偏空",
                        "confidence": 90,
                        "base_weight": 0.15,
                        "effective_weight": 0.15,
                        "detail": "8 条 · 官方占比 50%",
                    },
                ],
                "conflicts": [],
                "missing_components": [],
                "summary": (
                    "本地研判评分 13.6/100，方向明显偏空（强度 72.8），"
                    "可信度 89%；四类证据均已纳入并按各自可靠度加权。"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
