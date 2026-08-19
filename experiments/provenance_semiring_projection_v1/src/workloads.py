from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = EXPERIMENT_ROOT / "fixtures" / "workloads.json"
SUPPORTED_OPERATORS = {"base", "select", "project", "rename", "union", "join"}


def _stress_relations(rows_per_relation: int, key_modulus: int) -> dict[str, list[dict[str, Any]]]:
    if rows_per_relation < 50 or key_modulus < 2:
        raise ValueError("stress fixture must retain at least 50 rows per relation and two keys")
    relations: dict[str, list[dict[str, Any]]] = {"R": [], "S": []}
    for relation in ("R", "S"):
        for index in range(rows_per_relation):
            relations[relation].append(
                {
                    "source_identity": f"W12:{relation}:{relation.lower()}{index:03d}",
                    "values": {
                        "key": index % key_modulus,
                        "payload": f"{relation.lower()}-{index:03d}",
                    },
                }
            )
    return relations


def _walk_ast(node: dict[str, Any], stages: set[str]) -> None:
    operator = node.get("op")
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported positive-RA operator: {operator!r}")
    if operator != "base":
        stage = node.get("stage")
        if not isinstance(stage, str) or not stage or stage in stages:
            raise ValueError(f"stage must be a unique non-empty string: {stage!r}")
        stages.add(stage)
    if operator in {"select", "project", "rename"}:
        _walk_ast(node["input"], stages)
    elif operator == "union":
        if len(node.get("inputs", [])) < 2:
            raise ValueError("union requires at least two inputs")
        for child in node["inputs"]:
            _walk_ast(child, stages)
    elif operator == "join":
        _walk_ast(node["left"], stages)
        _walk_ast(node["right"], stages)


def load_workloads() -> list[dict[str, Any]]:
    document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if document.get("schema_version") != "positive-ra-workloads-v1":
        raise ValueError("unexpected workload schema version")
    workloads = deepcopy(document["workloads"])
    ids = [item.get("id") for item in workloads]
    if ids != [f"W{index}" for index in range(1, 13)]:
        raise ValueError("fixtures must contain exactly W1 through W12 in order")
    all_source_identities: set[str] = set()
    for workload in workloads:
        generator = workload.pop("relation_generator", None)
        if generator is not None:
            if generator.get("kind") != "stress_tables_v1":
                raise ValueError("unknown relation generator")
            workload["relations"] = _stress_relations(
                int(generator["rows_per_relation"]), int(generator["key_modulus"])
            )
            workload["relation_generator"] = generator
        queries = workload.get("queries") or {"default": workload["query"]}
        for query in queries.values():
            _walk_ast(query, set())
        local_identities: set[str] = set()
        for rows in workload["relations"].values():
            for row in rows:
                identity = row.get("source_identity")
                if not isinstance(identity, str) or not identity:
                    raise ValueError("every source row requires an explicit identity")
                if identity in local_identities or identity in all_source_identities:
                    raise ValueError(f"duplicate source identity: {identity}")
                local_identities.add(identity)
                all_source_identities.add(identity)
                if not isinstance(row.get("values"), dict):
                    raise ValueError("source values must be an object")
    return workloads


def select_query(workload: dict[str, Any], variant: str | None = None) -> tuple[str, dict[str, Any]]:
    if "queries" not in workload:
        if variant not in (None, "default"):
            raise ValueError(f"{workload['id']} has no query variant {variant!r}")
        return "default", workload["query"]
    selected = variant or workload["default_query"]
    try:
        return selected, workload["queries"][selected]
    except KeyError as exc:
        raise ValueError(f"unknown query variant {selected!r}") from exc


def workload_by_id(workload_id: str) -> dict[str, Any]:
    for workload in load_workloads():
        if workload["id"] == workload_id:
            return workload
    raise ValueError(f"unknown workload: {workload_id}")

