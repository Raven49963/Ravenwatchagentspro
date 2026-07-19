from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_starter import __version__
from quant_starter.global_market import USER_AGENT as GLOBAL_USER_AGENT
from quant_starter.instrument_catalog import CATALOG_USER_AGENT
from quant_starter.llm_client import PRODUCT_USER_AGENT as LLM_USER_AGENT
from quant_starter.metadata import APP_NAME, APP_VERSION, PRODUCT_USER_AGENT
from quant_starter.news import SEC_USER_AGENT, USER_AGENT as NEWS_USER_AGENT
from quant_starter.polymarket import USER_AGENT as POLYMARKET_USER_AGENT
from web_app import app


class ApplicationMetadataTests(unittest.TestCase):
    def test_public_versions_share_one_source(self) -> None:
        self.assertEqual(__version__, APP_VERSION)
        self.assertEqual(app.version, APP_VERSION)
        self.assertEqual(app.title, f"{APP_NAME} API")

        index_html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn(f"v{APP_VERSION} Polymarket", index_html)

    def test_outbound_clients_identify_the_same_release(self) -> None:
        self.assertEqual(LLM_USER_AGENT, PRODUCT_USER_AGENT)
        for user_agent in (
            GLOBAL_USER_AGENT,
            CATALOG_USER_AGENT,
            NEWS_USER_AGENT,
            SEC_USER_AGENT,
            POLYMARKET_USER_AGENT,
        ):
            self.assertIn(PRODUCT_USER_AGENT, user_agent)


if __name__ == "__main__":
    unittest.main()
