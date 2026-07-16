from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_app import QuantStarterApp


class QuantAppGuiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = QuantStarterApp(self.root)

    def tearDown(self) -> None:
        self.app._clear_kline()
        self.root.update_idletasks()
        self.root.destroy()

    def test_stock_selection_builds_embedded_kline(self) -> None:
        self.app.source.set("demo")
        self.app.strategy.set("ma")
        self.app._apply_source_defaults()
        self.app.stock_choice.set("Growth 成长 · GROWTH")
        self.app._on_stock_selected()

        self.assertEqual(self.app.ticker.get(), "GROWTH")
        self.assertEqual(self.app.symbols.get(), "GROWTH")

        request = self.app._snapshot_kline_request()
        bars = self.app._load_selected_bars(request)
        self.app._on_kline_success(bars, request.symbol)
        self.root.update_idletasks()

        self.assertEqual(self.app.current_kline_symbol, "GROWTH")
        self.assertIsNotNone(self.app.kline_canvas)
        self.assertIsNotNone(self.app.kline_figure)
        self.assertEqual(len(self.app.kline_figure.axes), 2)
        self.assertEqual(len(self.app.kline_figure.axes[0].patches), 120)


if __name__ == "__main__":
    unittest.main()
