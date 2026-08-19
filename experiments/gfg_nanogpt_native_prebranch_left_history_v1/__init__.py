"""Prospective left-history branch analysis for the frozen nanoGPT response corpus."""

from typing import Any

__all__ = ["DEFAULT_REPORT_ROOT", "run"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
