from __future__ import annotations


class CoreV3Error(ValueError):
    """A fail-closed Core v3 error with a stable machine reason code."""

    def __init__(self, reason_code: str, detail: str | None = None) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if detail is None else f"{reason_code}:{detail}")


def fail(reason_code: str, detail: str | None = None) -> "None":
    raise CoreV3Error(reason_code, detail)
