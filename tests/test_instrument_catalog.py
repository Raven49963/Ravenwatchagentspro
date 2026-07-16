from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from quant_starter.instrument_catalog import (
    CatalogInstrument,
    InstrumentCatalogService,
    MarketCatalog,
    _parse_nasdaq_directory,
    catalog_service,
    write_catalog_snapshot,
)
from web_app import _data_quality_payload


class InstrumentCatalogParsingTests(unittest.TestCase):
    def test_nasdaq_directory_filters_test_issues_and_derivatives(self) -> None:
        body = b"\n".join(
            [
                b"Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
                b"AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
                b"FUND|Example Index ETF|G|N|N|100|Y|N",
                b"WTEST|Example Corp Warrant|S|N|N|100|N|N",
                b"ZTEST|Nasdaq Test Issue|Q|Y|N|100|N|N",
                b"File Creation Time: 0715202618:00|||||||",
            ]
        )
        items = _parse_nasdaq_directory(body, listed_on_nasdaq=True)
        self.assertEqual([item.symbol for item in items], ["AAPL", "FUND"])
        self.assertEqual(items[0].name, "Apple Inc.")
        self.assertEqual(items[1].asset_type, "etf")

    def test_other_exchange_class_share_uses_application_symbol(self) -> None:
        body = b"\n".join(
            [
                b"ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
                b"BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B",
            ]
        )
        items = _parse_nasdaq_directory(body, listed_on_nasdaq=False)
        self.assertEqual(items[0].symbol, "BRK-B")
        self.assertEqual(items[0].exchange, "NYSE")


class InstrumentCatalogQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        items = [
            CatalogInstrument(
                market="a-share",
                symbol=f"600{index:03d}",
                name=f"测试公司 {index}",
                asset_type="stock" if index < 110 else "etf",
                exchange="上交所",
                country="CN",
                currency="CNY",
                category="金融" if index % 2 == 0 else "工业",
                rank=index + 1,
            )
            for index in range(125)
        ]
        catalog = MarketCatalog(
            market="a-share",
            items=tuple(items),
            provider="test-catalog",
            provider_label="测试目录",
            provider_url="https://example.com/catalog",
            updated_at="2026-07-15T10:00:00+00:00",
            source_mode="online",
        )
        snapshot = {
            "schema_version": 1,
            "generated_at": catalog.updated_at,
            "markets": {"a-share": catalog.to_storage_dict()},
        }
        snapshot_path = root / "catalog.json.gz"
        write_catalog_snapshot(snapshot, snapshot_path)
        self.service = InstrumentCatalogService(
            bundled_path=snapshot_path,
            cache_dir=root / "cache",
            auto_refresh=False,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_catalog_paginates_and_exposes_counts(self) -> None:
        payload = self.service.query(
            market="a-share", page=2, page_size=50
        )
        self.assertEqual(payload["catalog_total"], 125)
        self.assertEqual(payload["filtered_total"], 125)
        self.assertEqual(payload["pages"], 3)
        self.assertEqual(len(payload["items"]), 50)
        self.assertEqual(payload["items"][0]["symbol"], "600050")
        self.assertEqual(payload["asset_counts"]["etf"], 15)

    def test_catalog_combines_asset_category_and_keyword_filters(self) -> None:
        payload = self.service.query(
            market="a-share",
            q="测试公司 11",
            asset_type="etf",
            category="金融",
            page_size=50,
        )
        self.assertTrue(payload["items"])
        self.assertTrue(all(item["asset_type"] == "etf" for item in payload["items"]))
        self.assertTrue(all(item["category"] == "金融" for item in payload["items"]))


class BundledCatalogCoverageTests(unittest.TestCase):
    def test_bundled_catalog_has_large_multi_market_universe(self) -> None:
        self.assertGreaterEqual(catalog_service.summary("a-share")["count"], 5_000)
        self.assertGreaterEqual(catalog_service.summary("nasdaq")["count"], 10_000)
        self.assertGreaterEqual(catalog_service.summary("hk")["count"], 2_500)
        self.assertGreaterEqual(catalog_service.summary("global")["count"], 15)

    def test_bundled_catalog_finds_exact_codes(self) -> None:
        a_share = catalog_service.query(market="a-share", q="600519", page_size=20)
        us = catalog_service.query(market="nasdaq", q="BRK-B", page_size=20)
        hk = catalog_service.query(market="hk", q="0700.HK", page_size=20)
        self.assertEqual(a_share["items"][0]["symbol"], "600519")
        self.assertEqual(us["items"][0]["symbol"], "BRK-B")
        self.assertEqual(hk["items"][0]["symbol"], "0700.HK")
        padded_hk = catalog_service.query(market="hk", q="00005", page_size=20)
        dotted_us = catalog_service.query(market="nasdaq", q="BRK.B", page_size=20)
        self.assertEqual(padded_hk["items"][0]["symbol"], "0005.HK")
        self.assertEqual(dotted_us["items"][0]["symbol"], "BRK-B")

    def test_bundled_catalog_prioritizes_common_research_targets(self) -> None:
        self.assertEqual(
            catalog_service.query(market="a-share", page_size=20)["items"][0]["symbol"],
            "600519",
        )
        self.assertEqual(
            catalog_service.query(market="nasdaq", page_size=20)["items"][0]["symbol"],
            "AAPL",
        )
        self.assertEqual(
            catalog_service.query(market="hk", page_size=20)["items"][0]["symbol"],
            "0700.HK",
        )


class DataQualityTests(unittest.TestCase):
    def test_data_quality_reports_coverage_and_invalid_rows(self) -> None:
        dates = pd.to_datetime(["2026-07-13", "2026-07-14", "2026-07-14"])
        frame = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 11.0],
                "High": [11.0, 10.0, 12.0],
                "Low": [9.0, 10.5, 10.0],
                "Close": [10.5, 11.5, 11.8],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )
        quality = _data_quality_payload(frame, period="1y", adjustment="qfq")
        self.assertEqual(quality["rows"], 3)
        self.assertEqual(quality["duplicate_dates"], 1)
        self.assertEqual(quality["invalid_rows"], 1)
        self.assertEqual(quality["status"], "不足")
        self.assertLess(quality["score"], 100)


if __name__ == "__main__":
    unittest.main()
