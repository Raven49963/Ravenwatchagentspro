from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.global_market import (
    MSNInstrument,
    _range_code,
    fetch_msn_ohlcv,
    resolve_msn_instrument,
)
from quant_starter.news import _fetch_hkex_announcements, _fetch_sec_filings
from quant_starter.realtime import fetch_realtime_market
from quant_starter.symbols import presets_for_source, resolve_stock_choice
from web_app import markets, search_instruments, symbols


def instrument(
    symbol: str,
    instrument_id: str,
    *,
    name: str = "Test Security",
    country: str = "JP",
    mic: str = "XTKS",
    asset_type: str = "stock",
) -> MSNInstrument:
    return MSNInstrument(
        symbol=symbol,
        name=name,
        instrument_id=instrument_id,
        full_instrument=f"test.{symbol}",
        market="global",
        exchange="Tokyo",
        mic=mic,
        country=country,
        asset_type=asset_type,
        currency="JPY",
        timezone="Asia/Tokyo",
    )


class GlobalMarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        from quant_starter import global_market

        global_market._resolve_cache.clear()
        global_market._search_cache.clear()

    def test_microsoft_range_uses_supported_chart_periods(self) -> None:
        self.assertEqual(_range_code("2026-04-01", "2026-07-15"), "3M")
        self.assertEqual(_range_code("2026-01-15", "2026-07-15"), "1Y")
        self.assertEqual(_range_code("2021-07-15", "2026-07-15"), "5Y")

    def test_resolver_prefers_exact_symbol_before_exchange_hint(self) -> None:
        exact_index = instrument(
            "N225.T",
            "index-id",
            name="Nikkei 225 Index",
            mic="XTKS_N225",
            asset_type="index",
        )
        exchange_match = instrument(
            "1369.T",
            "etf-id",
            name="Nikkei ETF",
            mic="XTKS",
            asset_type="etf",
        )
        with mock.patch(
            "quant_starter.global_market.search_msn_instruments",
            return_value=(exchange_match, exact_index),
        ):
            selected = resolve_msn_instrument("N225.T", "global")
        self.assertEqual(selected.instrument_id, "index-id")
        self.assertEqual(selected.asset_type, "index")

    def test_resolver_accepts_class_share_separator_aliases(self) -> None:
        berkshire = instrument(
            "BRK.B",
            "berkshire-b",
            name="Berkshire Hathaway Inc",
            country="US",
            mic="XNYS",
        )
        with mock.patch(
            "quant_starter.global_market.search_msn_instruments",
            return_value=(berkshire,),
        ):
            selected = resolve_msn_instrument("BRK-B", "nasdaq")
        self.assertEqual(selected.instrument_id, "berkshire-b")
        self.assertEqual(selected.symbol, "BRK.B")

    def test_global_ohlcv_normalizes_chart_and_provenance(self) -> None:
        selected = instrument("7203.T", "toyota-id", name="Toyota Motor")
        chart = {
            "series": {
                "timeStamps": [
                    "2026-01-15T06:00:00Z",
                    "2026-04-15T06:00:00Z",
                    "2026-07-15T06:00:00Z",
                ],
                "openPrices": [2800, 2850, 2900],
                "pricesHigh": [2820, 2870, 2920],
                "pricesLow": [2780, 2830, 2880],
                "prices": [2810, 2860, 2910],
                "volumes": [1000, 1200, 1300],
            }
        }
        with (
            mock.patch(
                "quant_starter.global_market.resolve_msn_instrument",
                return_value=selected,
            ),
            mock.patch(
                "quant_starter.global_market.fetch_msn_chart",
                return_value=chart,
            ) as chart_loader,
        ):
            frame = fetch_msn_ohlcv(
                "7203.T", "global", "2026-01-15", "2026-07-15"
            )
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.attrs["provider"], "msn-finance")
        self.assertTrue(frame.attrs["provider_url"].endswith("fi-toyotaid"))
        chart_loader.assert_called_once_with(selected, "1Y")


class MultiMarketMetadataTests(unittest.TestCase):
    @staticmethod
    def _hk_quote_body() -> bytes:
        fields = [""] * 80
        fields[1] = "XL二南方恒科"
        fields[2] = "07226"
        fields[3] = "3.49"
        fields[4] = "3.40"
        fields[5] = "3.42"
        fields[6] = "120000"
        fields[30] = "2026/07/15 15:30:00"
        fields[33] = "3.51"
        fields[34] = "3.38"
        fields[35] = "3.49/120000/418800"
        return ('v_r_hk07226="' + "~".join(fields) + '";').encode("gb18030")

    @staticmethod
    def _hk_minute_body() -> bytes:
        payload = {
            "code": 0,
            "data": {
                "hk07226": {
                    "data": {
                        "date": "20260715",
                        "data": [
                            "0930 3.42 1000 3420",
                            "0931 3.49 1800 6282",
                        ],
                    }
                }
            },
        }
        return json.dumps(payload).encode("utf-8")

    def test_presets_cover_four_markets_and_etfs(self) -> None:
        self.assertGreaterEqual(len(presets_for_source("a-share", "etf")), 9)
        self.assertGreaterEqual(len(presets_for_source("nasdaq", "etf")), 15)
        self.assertGreaterEqual(len(presets_for_source("hk", "etf")), 7)
        self.assertGreaterEqual(len(presets_for_source("global", "etf")), 4)
        self.assertEqual(resolve_stock_choice("700", "hk"), "0700.HK")
        self.assertEqual(resolve_stock_choice("CSPX.L", "global"), "CSPX.L")

    def test_hk_leveraged_etf_uses_catalog_metadata(self) -> None:
        def response(request, **_kwargs):
            if "minute/query" in request.full_url:
                return self._hk_minute_body()
            return self._hk_quote_body()

        with mock.patch(
            "quant_starter.realtime._read_http_bytes", side_effect=response
        ):
            live = fetch_realtime_market("hk", "7226.HK")
        self.assertEqual(live.snapshot.asset_type, "etf")
        self.assertEqual(live.snapshot.asset_type_label, "ETF")
        self.assertEqual(live.snapshot.exchange, "港交所")
        self.assertEqual(live.snapshot.currency, "HKD")


class TrustedNewsSourceTests(unittest.TestCase):
    def test_sec_filing_has_direct_document_and_regulator_metadata(self) -> None:
        payload = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "accessionNumber": ["0001045810-26-000001"],
                    "primaryDocument": ["nvda-20260715.htm"],
                    "primaryDocDescription": ["Quarterly report"],
                    "acceptanceDateTime": ["2026-07-15T10:30:00Z"],
                    "filingDate": ["2026-07-15"],
                }
            }
        }
        with (
            mock.patch(
                "quant_starter.news._sec_company_map",
                return_value={"NVDA": (1045810, "NVIDIA CORP")},
            ),
            mock.patch(
                "quant_starter.news._read_bytes",
                return_value=json.dumps(payload).encode("utf-8"),
            ),
        ):
            items = _fetch_sec_filings("NVDA", 5, 5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_kind, "filing")
        self.assertEqual(items[0].credibility, "监管机构官方")
        self.assertIn("/Archives/edgar/data/1045810/", items[0].url)

    def test_hkex_announcement_has_direct_pdf_and_exchange_metadata(self) -> None:
        html = b"""
        <table><tr>
          <td class="release-time">Release Time: 15/07/2026 08:30</td>
          <td><div class="headline">Results Announcements</div>
          <div class="doc-link"><a href="/listedco/listconews/sehk/2026/0715/report.pdf">Interim Results</a></div></td>
        </tr></table>
        """
        with (
            mock.patch(
                "quant_starter.news._hkex_company_map",
                return_value={"00700": (7609, "TENCENT")},
            ),
            mock.patch("quant_starter.news._read_bytes", return_value=html),
        ):
            items = _fetch_hkex_announcements("0700.HK", 5, 5)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].url.endswith("report.pdf"))
        self.assertEqual(items[0].credibility, "交易所官方披露")
        self.assertEqual(items[0].publisher, "HKEXnews")


class MarketApiTests(unittest.TestCase):
    def test_market_and_symbol_catalog_expose_asset_metadata(self) -> None:
        market_payload = markets()
        self.assertEqual(
            [item["id"] for item in market_payload["markets"]],
            ["a-share", "nasdaq", "hk", "global"],
        )
        symbol_payload = symbols(market="hk", asset_type="etf")
        self.assertTrue(symbol_payload["symbols"])
        self.assertTrue(
            all(item["asset_type"] == "etf" for item in symbol_payload["symbols"])
        )
        self.assertTrue(all(item["exchange"] for item in symbol_payload["symbols"]))

    def test_online_instrument_search_serializes_source_link(self) -> None:
        remote = instrument(
            "TYT.L",
            "toyota-london",
            name="Toyota Motor Corp",
            country="GB",
            mic="XLON",
        )
        with mock.patch("web_app.search_msn_instruments", return_value=(remote,)):
            payload = asyncio.run(
                search_instruments(
                    q="TYT",
                    market="global",
                    asset_type="all",
                    limit=12,
                )
            )
        self.assertTrue(payload["online"])
        self.assertEqual(payload["items"][0]["symbol"], "TYT.L")
        self.assertTrue(payload["items"][0]["source_url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
