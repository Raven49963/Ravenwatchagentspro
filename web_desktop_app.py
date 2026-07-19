from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
from urllib.parse import quote
from urllib.request import urlopen

import uvicorn

from web_app import app
from quant_starter.metadata import APP_NAME, PRODUCT_ID


def _resource_path(relative_path: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / relative_path


def _runtime_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", Path.home()))
    path = base / PRODUCT_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_logging() -> None:
    log_dir = _runtime_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "desktop.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class DesktopServer:
    def __init__(self, port: int, token: str) -> None:
        self.port = port
        self.token = token
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            log_config=None,
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run,
            name="ravenwatchagents-api",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/api/health?desktop_token={quote(self.token)}"

    def start(self, timeout: float = 20) -> dict[str, object]:
        self.thread.start()
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if not self.thread.is_alive():
                raise RuntimeError("本地行情服务启动失败。")
            try:
                with urlopen(self.health_url, timeout=2) as response:
                    payload = json.load(response)
                logging.info("Desktop API ready on %s", self.base_url)
                return payload
            except Exception as exc:
                last_error = exc
                time.sleep(0.15)
        raise RuntimeError(f"本地行情服务启动超时：{last_error}")

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} Desktop")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--window-smoke-test", action="store_true")
    return parser.parse_args()


def _verify_and_close_window(window, result: dict[str, object]) -> None:
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    try:
        while time.monotonic() < deadline:
            try:
                ready = window.evaluate_js(
                    "document.readyState === 'complete' "
                    "&& Boolean(window.echarts) "
                    "&& Boolean(window.lucide) "
                    "&& Boolean(document.querySelector('#kline-chart canvas')) "
                    "&& document.querySelectorAll('.evidence-component-row').length === 5 "
                    "&& document.querySelectorAll('#deliberation-process-list li').length === 5"
                )
                if ready:
                    result["ready"] = True
                    result["title"] = window.evaluate_js("document.title")
                    logging.info("Desktop window smoke test passed: %s", result["title"])
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
        result["error"] = str(last_error or "窗口内容加载超时")
    finally:
        window.destroy()


def main() -> None:
    _configure_logging()
    args = parse_args()
    token = secrets.token_urlsafe(32)
    os.environ["RAVENWATCHAGENTS_DESKTOP_TOKEN"] = token
    server = DesktopServer(args.port or _find_free_port(), token)
    try:
        health = server.start()
        if args.smoke_test:
            url = (
                f"{server.base_url}/api/polymarket?market=nasdaq&symbol=NVDA"
                f"&offline=true&desktop_token={quote(token)}"
            )
            with urlopen(url, timeout=10) as response:
                prediction_payload = json.load(response)
            process = prediction_payload.get("assessment", {}).get("process", [])
            if len(process) != 5:
                raise RuntimeError("桌面包内 Polymarket 五步研判接口不完整。")
            logging.info(
                "Smoke test passed: health=%s polymarket=%s steps=%s",
                health,
                prediction_payload.get("source_mode"),
                len(process),
            )
            return

        import webview

        storage_path = _runtime_dir() / "webview"
        storage_path.mkdir(parents=True, exist_ok=True)
        url = f"{server.base_url}/?desktop_token={quote(token)}"
        window = webview.create_window(
            f"{APP_NAME} 研投终端",
            url=url,
            width=1440,
            height=940,
            min_size=(1024, 700),
            resizable=True,
            background_color="#080d0f",
            text_select=True,
        )
        smoke_result: dict[str, object] = {"ready": False}
        webview.start(
            _verify_and_close_window if args.window_smoke_test else None,
            (window, smoke_result) if args.window_smoke_test else None,
            debug=False,
            private_mode=False,
            storage_path=str(storage_path),
            icon=str(_resource_path("web/ravenwatchagentspro.ico")),
        )
        if args.window_smoke_test and not smoke_result["ready"]:
            raise RuntimeError(f"桌面窗口验收失败：{smoke_result.get('error', '未知错误')}")
    except Exception:
        logging.exception("Desktop application failed")
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
