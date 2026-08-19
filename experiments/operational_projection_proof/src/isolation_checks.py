from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable

from generation_relation_core.snapshots import AUTHORITATIVE_TABLE_SPECS


DOMAIN_AUTHORITY_TOKENS = (
    "lineage_table",
    "lineage_from",
    "lineage_to",
    "input_output_map",
    "provenance_circuit",
    "span_table",
    "source_map_table",
)
NATIVE_READ_METHODS = frozenset({"open", "read_text", "read_bytes"})


def _imports(tree: ast.AST) -> list[str]:
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result.append(module)
            result.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    return result


def scan_candidate_isolation(
    paths: Iterable[Path], *, forbidden_modules: Iterable[str]
) -> dict:
    forbidden = tuple(forbidden_modules)
    oracle_leaks: list[str] = []
    native_reads: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for imported in _imports(tree):
            if any(token in imported for token in forbidden):
                oracle_leaks.append(f"{path.name}:{imported}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in NATIVE_READ_METHODS:
                    native_reads.append(f"{path.name}:{name}")
    return {
        "candidate_files": sorted(path.name for path in paths),
        "oracle_leakage_count": len(oracle_leaks),
        "oracle_leakage_findings": sorted(oracle_leaks),
        "native_domain_result_read_count": len(native_reads),
        "native_domain_result_read_findings": sorted(native_reads),
        "status": "SUPPORTED"
        if not oracle_leaks and not native_reads
        else "NOT_SUPPORTED",
    }


def scan_source_isolation(
    source: str, *, filename: str, forbidden_modules: Iterable[str]
) -> dict:
    tree = ast.parse(source, filename=filename)
    forbidden = tuple(forbidden_modules)
    oracle_leaks = [
        f"{filename}:{imported}"
        for imported in _imports(tree)
        if any(token in imported for token in forbidden)
    ]
    native_reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        )
        if name in NATIVE_READ_METHODS:
            native_reads.append(f"{filename}:{name}")
    return {
        "candidate_files": [filename],
        "oracle_leakage_count": len(oracle_leaks),
        "oracle_leakage_findings": sorted(oracle_leaks),
        "native_domain_result_read_count": len(native_reads),
        "native_domain_result_read_findings": sorted(native_reads),
        "status": "SUPPORTED"
        if not oracle_leaks and not native_reads
        else "NOT_SUPPORTED",
    }


def require_candidate_isolation(report: dict) -> None:
    if report.get("oracle_leakage_count"):
        from .errors import ProjectionProofError

        raise ProjectionProofError("ORACLE_LEAKAGE")
    if report.get("native_domain_result_read_count"):
        from .errors import ProjectionProofError

        raise ProjectionProofError("NATIVE_DOMAIN_RESULT_LEAKAGE")


def detect_second_authority_source(source: str) -> list[str]:
    tree = ast.parse(source)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and any(
                token in target.id.lower() for token in DOMAIN_AUTHORITY_TOKENS
            ):
                findings.append(target.id)
    return sorted(set(findings))


def require_no_second_authority_source(source: str) -> None:
    findings = detect_second_authority_source(source)
    if findings:
        from .errors import ProjectionProofError

        raise ProjectionProofError("SECOND_AUTHORITY_STORE", ",".join(findings))


def _json_property_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for child in value.values():
            names.update(_json_property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_json_property_names(child))
    return names


def audit_second_authority(
    *,
    schema_path: Path,
    candidate_paths: Iterable[Path],
    expected_schema_sha256: str,
) -> dict:
    schema_bytes = schema_path.read_bytes()
    schema_hash = hashlib.sha256(schema_bytes).hexdigest()
    schema = json.loads(schema_bytes)
    names = set(AUTHORITATIVE_TABLE_SPECS) | _json_property_names(schema)
    core_findings = sorted(
        name
        for name in names
        if any(token in name.lower() for token in DOMAIN_AUTHORITY_TOKENS)
    )
    source_findings: list[str] = []
    for path in candidate_paths:
        for name in detect_second_authority_source(path.read_text(encoding="utf-8")):
            source_findings.append(f"{path.name}:{name}")
    findings = [*core_findings, *source_findings]
    schema_unchanged = schema_hash == expected_schema_sha256
    return {
        "authoritative_core_tables": sorted(AUTHORITATIVE_TABLE_SPECS),
        "core_schema_sha256": schema_hash,
        "expected_frozen_schema_sha256": expected_schema_sha256,
        "core_schema_unchanged": schema_unchanged,
        "second_authority_store_count": len(findings),
        "findings": findings,
        "transient_indexes_are_non_authoritative": True,
        "status": "SUPPORTED" if schema_unchanged and not findings else "NOT_SUPPORTED",
    }
