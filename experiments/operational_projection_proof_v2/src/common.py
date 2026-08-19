from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from generation_relation_core.canonical import canonical_bytes


class ProofFailure(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


def require(condition: bool, reason_code: str, detail: str = "") -> None:
    if not condition:
        raise ProofFailure(reason_code, detail)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if check and result.returncode:
        raise ProofFailure("GIT_COMMAND_FAILED", f"{' '.join(args)}:{result.stderr.strip()}")
    return result.stdout.strip()


def git_exit(repo_root: Path, *args: str) -> int:
    return subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True
    ).returncode


def snapshot_document(snapshot: Any) -> dict[str, Any]:
    return {
        "snapshot": snapshot.record,
        "tables": {
            field: getattr(snapshot.tables, field)
            for field in snapshot.tables.__dataclass_fields__
        },
    }


def canonical_row_set(rows: Iterable[dict[str, Any]]) -> set[bytes]:
    return {canonical_bytes(row) for row in rows}


def set_comparison(
    left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    left_set = canonical_row_set(left)
    right_set = canonical_row_set(right)
    difference = left_set ^ right_set
    return {
        "equal": left_set == right_set,
        "left_count": len(left_set),
        "right_count": len(right_set),
        "symmetric_difference_count": len(difference),
        "left_only_sha256": sha256_bytes(b"".join(sorted(left_set - right_set))),
        "right_only_sha256": sha256_bytes(b"".join(sorted(right_set - left_set))),
    }


def text_set_comparison(left: Iterable[str], right: Iterable[str]) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    return {
        "equal": left_set == right_set,
        "left_count": len(left_set),
        "right_count": len(right_set),
        "symmetric_difference_count": len(left_set ^ right_set),
        "left_only": sorted(left_set - right_set),
        "right_only": sorted(right_set - left_set),
    }


def tree_sha256(path: Path, *, excluded_parts: frozenset[str] = frozenset()) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if not item.is_file() or any(part in excluded_parts for part in item.parts):
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_hashes(path: Path, *, repo_root: Path) -> dict[str, str]:
    return {
        item.relative_to(repo_root).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*.py"), key=lambda value: value.as_posix())
        if "__pycache__" not in item.parts
    }


def without_nondeterministic_formal_fields(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"performance_seconds", "peak_process_rss_bytes"}
    }

