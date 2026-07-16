from __future__ import annotations

import sys


class NullTextStream:
    """Minimal text stream for third-party libraries in a windowed executable."""

    encoding = "utf-8"

    def write(self, value: object) -> int:
        return len(str(value))

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


def ensure_gui_streams() -> None:
    if sys.stdout is None:
        sys.stdout = NullTextStream()
    if sys.stderr is None:
        sys.stderr = NullTextStream()
