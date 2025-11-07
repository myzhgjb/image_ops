"""Runtime helpers to ensure UTF-8 compatibility on Windows consoles.

The project relies heavily on Chinese UI strings and console outputs.
Without extra handling, Windows' default code page (often GBK/CP936)
will render those strings as mojibake when printing to stdout/stderr.
This module provides a single helper that configures Python's IO layer
to UTF-8 and switches the console code page when possible.
"""

from __future__ import annotations

import locale
import os
import sys
from typing import Optional


def _try_set_console_cp() -> None:
    """Switch Windows console to UTF-8 if available.

    Uses Win32 APIs via ctypes. If anything fails we silently ignore it so
    the application can continue to run without crashing.
    """

    if sys.platform != "win32":
        return

    try:  # pragma: no cover - Windows specific, hard to test automatically
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        CP_UTF8 = 65001
        kernel32.SetConsoleCP(CP_UTF8)
        kernel32.SetConsoleOutputCP(CP_UTF8)
    except Exception:
        # Ignore any failure (e.g. running in an environment without a console)
        pass


def _reconfigure_stream(stream: Optional[object]) -> None:
    """Force a text stream to UTF-8 if Python exposes ``reconfigure``."""

    if stream is None:
        return

    try:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_utf8_console() -> None:
    """Best-effort UTF-8 setup for stdout/stderr and locale.

    Should be called as early as possible in entry scripts to avoid
    mojibake for both current and subsequent log/output lines.
    """

    # Respect explicit user configuration.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if sys.platform == "win32":
        # Ensure the process prefers UTF-8 for locale aware operations.
        try:
            os.environ.setdefault("LC_ALL", "")
            os.environ.setdefault("LANG", "zh_CN.UTF-8")
        except Exception:
            pass

        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            # Fall back to a UTF-8 locale if available.
            for candidate in ("zh_CN.UTF-8", "zh_CN", "en_US.UTF-8"):
                try:
                    locale.setlocale(locale.LC_ALL, candidate)
                    break
                except locale.Error:
                    continue

        _try_set_console_cp()

    _reconfigure_stream(getattr(sys, "stdout", None))
    _reconfigure_stream(getattr(sys, "stderr", None))


__all__ = ["ensure_utf8_console"]


