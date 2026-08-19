from __future__ import annotations

import ast
from pathlib import Path

from .projection_errors import ProjectionError


PROHIBITED_DIRECT_DEPENDENCIES = (
    "native_otel_capture",
    "independent_oracle",
    ".tests",
    ".artifacts",
)


def prohibited_imports_in_source(source: str) -> list[str]:
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return sorted(
        name
        for name in imports
        if any(token in name for token in PROHIBITED_DIRECT_DEPENDENCIES)
    )


def assert_static_projection_isolation(paths: list[Path]) -> None:
    violations = []
    for path in paths:
        for imported in prohibited_imports_in_source(path.read_text(encoding="utf-8")):
            violations.append(f"{path.name}:{imported}")
    if violations:
        raise ProjectionError(
            "ORACLE_OR_NATIVE_DEPENDENCY_PROHIBITED", ",".join(violations)
        )


def assert_injected_dependency_rejected() -> str:
    injected = "from .native_otel_capture import NativeOtelCapture\n"
    if not prohibited_imports_in_source(injected):
        raise AssertionError("isolation checker failed to detect injected dependency")
    try:
        raise ProjectionError("ORACLE_OR_NATIVE_DEPENDENCY_PROHIBITED", "INJECTED")
    except ProjectionError as exc:
        return exc.reason_code


def count_otel_core_fields(repository_root: Path) -> int:
    terms = ("span_id", "trace_id", "otel_parent", "opentelemetry")
    count = 0
    roots = [
        repository_root / "protocol" / "core_v3",
        repository_root / "src" / "generation_relation_core",
    ]
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                text = path.read_text(encoding="utf-8").lower()
                count += sum(text.count(term) for term in terms)
    return count
