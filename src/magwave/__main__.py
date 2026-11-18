"""``python -m magwave`` — the headless entry point CI uses (FR-060)."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
