from __future__ import annotations


class ProjectionProofError(ValueError):
    """A machine-readable proof failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
