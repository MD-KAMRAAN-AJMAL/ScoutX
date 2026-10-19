"""Source-checkout package shim delegating to root __init__.py."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(_ROOT)]

_ROOT_INIT = _ROOT / "__init__.py"
exec(compile(_ROOT_INIT.read_text(encoding="utf-8"), str(_ROOT_INIT), "exec"), globals())
