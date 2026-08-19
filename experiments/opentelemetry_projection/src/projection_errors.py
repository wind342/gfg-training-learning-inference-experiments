from __future__ import annotations


class ProjectionError(ValueError):
    """A fail-closed projection or comparison failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)
