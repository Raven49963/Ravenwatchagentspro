# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata


project_root = Path(SPECPATH)
datas = [
    (str(project_root / "web"), "web"),
    (
        str(project_root / "src" / "quant_starter" / "resources"),
        "quant_starter/resources",
    ),
]
hiddenimports = []

for package in ("akshare", "yfinance", "curl_cffi", "fastapi", "uvicorn", "webview"):
    datas += collect_data_files(package)
    hiddenimports += collect_submodules(package)

datas += copy_metadata("pywebview")


a = Analysis(
    ["web_desktop_app.py"],
    pathex=[str(project_root), str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "gi"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RavenWatchAgentsPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_root / "web" / "ravenwatchagentspro.ico")],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RavenWatchAgentsPro",
)
