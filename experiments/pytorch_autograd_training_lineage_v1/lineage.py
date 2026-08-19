from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot


_ORDINAL = re.compile(r"(?:^|\|)ordinal=(\d+)(?:\||$)")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class TrainingLineageIndex:
    """Bidirectional traversal over validated Core relations only."""

    def __init__(self, snapshot: ValidatedSnapshot, validation: SnapshotValidation) -> None:
        if validation.snapshot_id != snapshot.snapshot_id:
            raise ValueError("LINEAGE_SNAPSHOT_VALIDATION_MISMATCH")
        self.snapshot = snapshot
        tables = snapshot.tables
        self._sources = {
            row["source_information_id"]: row for row in tables.source_information_records
        }
        self._supports = {
            row["support_id"]: row for row in tables.perceptual_support_records
        }
        self._occurrences = {
            row["generation_occurrence_id"]: row for row in tables.generation_occurrences
        }
        generated = {
            row["generated_origin_id"]: row for row in tables.generated_origins
        }
        self._semantic_keys = {
            **{
                row_id: row["source_payload"]["source_ref"]
                for row_id, row in self._sources.items()
            },
            **{
                row_id: row["support_payload"]["support_key"]
                for row_id, row in self._supports.items()
            },
        }
        self._incoming: dict[str, list[dict[str, Any]]] = {}
        self._outgoing: dict[str, list[dict[str, Any]]] = {}
        self._context_occurrences_by_outcome: dict[str, set[str]] = {}
        for binding in tables.generation_bindings:
            origin = binding["origin_reference"]
            if origin["kind"] == "registered_source":
                origin_id = origin["source_information_id"]
            else:
                origin_id = generated[origin["generated_origin_id"]]["origin_payload"]["source_support_id"]
            outcome = binding["outcome_reference"]
            if outcome["kind"] != "support":
                continue
            outcome_id = outcome["support_id"]
            occurrence = self._occurrences[binding["generation_occurrence_id"]]
            if binding["relation_role"].startswith("backward_invocation_context|"):
                backward_id = occurrence["occurrence_payload"].get("backward_occurrence_id")
                if backward_id is not None:
                    self._context_occurrences_by_outcome.setdefault(outcome_id, set()).add(backward_id)
                continue
            ordinal_match = _ORDINAL.search(binding["relation_role"])
            edge = {
                "binding_id": binding["generation_binding_id"],
                "occurrence_id": occurrence["generation_occurrence_id"],
                "occurrence_key": occurrence["stable_instance_key"],
                "ordinal": None if ordinal_match is None else int(ordinal_match.group(1)),
                "origin_id": origin_id,
                "origin_key": self._semantic_keys[origin_id],
                "outcome_id": outcome_id,
                "outcome_key": self._semantic_keys[outcome_id],
                "role": binding["relation_role"],
            }
            self._incoming.setdefault(outcome_id, []).append(edge)
            self._outgoing.setdefault(origin_id, []).append(edge)
        for mapping in (self._incoming, self._outgoing):
            for key in mapping:
                mapping[key].sort(key=lambda row: row["binding_id"])

    def source_id_for_ref(self, source_ref: str) -> str:
        matches = [row_id for row_id, key in self._semantic_keys.items() if key == source_ref and row_id in self._sources]
        if len(matches) != 1:
            raise ValueError(f"SOURCE_REF_CARDINALITY:{source_ref}:{len(matches)}")
        return matches[0]

    def support_id_for_key(self, support_key: str) -> str:
        matches = [row_id for row_id, key in self._semantic_keys.items() if key == support_key and row_id in self._supports]
        if len(matches) != 1:
            raise ValueError(f"SUPPORT_KEY_CARDINALITY:{support_key}:{len(matches)}")
        return matches[0]

    def _reverse_paths(self, target_id: str, active: frozenset[str]) -> list[list[dict[str, Any]]]:
        if target_id in active:
            raise ValueError("LINEAGE_CYCLE")
        if target_id in self._sources:
            return [[]]
        edges = self._incoming.get(target_id, [])
        if not edges:
            raise ValueError(f"LINEAGE_ORIGIN_MISSING:{target_id}")
        result = []
        for edge in edges:
            prefixes = self._reverse_paths(edge["origin_id"], active | {target_id})
            result.extend([*prefix, edge] for prefix in prefixes)
        return result

    def reverse_lineage(self, target_result_id: str) -> dict[str, Any]:
        if target_result_id not in self._supports:
            raise ValueError("LINEAGE_TARGET_SUPPORT_UNKNOWN")
        paths = self._reverse_paths(target_result_id, frozenset())
        return self._format_query("reverse", target_result_id, paths)

    def _forward_paths(
        self,
        origin_id: str,
        prefix: list[dict[str, Any]],
        active: frozenset[str],
    ) -> list[list[dict[str, Any]]]:
        if origin_id in active:
            raise ValueError("LINEAGE_CYCLE")
        result = []
        for edge in self._outgoing.get(origin_id, []):
            path = [*prefix, edge]
            result.append(path)
            result.extend(self._forward_paths(edge["outcome_id"], path, active | {origin_id}))
        return result

    def forward_lineage(self, source_id: str) -> dict[str, Any]:
        if source_id not in self._sources:
            raise ValueError("LINEAGE_SOURCE_UNKNOWN")
        paths = self._forward_paths(source_id, [], frozenset())
        return self._format_query("forward", source_id, paths)

    def _format_query(
        self,
        direction: str,
        query_id: str,
        paths: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        semantic_paths = []
        multiplicities: dict[bytes, int] = {}
        path_by_key: dict[bytes, dict[str, Any]] = {}
        for path in paths:
            relations = [
                {
                    "occurrence_key": edge["occurrence_key"],
                    "ordinal": edge["ordinal"],
                    "origin_key": edge["origin_key"],
                    "outcome_key": edge["outcome_key"],
                    "role": edge["role"],
                }
                for edge in path
            ]
            semantic = {
                "relations": relations,
                "source_key": relations[0]["origin_key"],
                "target_key": relations[-1]["outcome_key"],
            }
            key = _canonical_bytes(semantic)
            multiplicities[key] = multiplicities.get(key, 0) + 1
            path_by_key[key] = semantic
        for key in sorted(path_by_key):
            semantic_paths.append({**path_by_key[key], "multiplicity": multiplicities[key]})
        relations = [edge for path in paths for edge in path]
        source_ids = sorted({path[0]["origin_id"] for path in paths if path})
        outcome_ids = sorted({edge["outcome_id"] for edge in relations})
        occurrence_ids_set = {edge["occurrence_id"] for edge in relations}
        for outcome_id in {edge["outcome_id"] for edge in relations}:
            occurrence_ids_set.update(self._context_occurrences_by_outcome.get(outcome_id, set()))
        occurrence_ids = sorted(occurrence_ids_set)
        roles = sorted({edge["role"] for edge in relations})
        payload = {
            "direction": direction,
            "occurrence_ids": occurrence_ids,
            "occurrence_keys": sorted({
                self._occurrences[occurrence_id]["stable_instance_key"]
                for occurrence_id in occurrence_ids
            }),
            "outcome_ids": outcome_ids,
            "outcome_keys": sorted({self._semantic_keys[row_id] for row_id in outcome_ids}),
            "path_count": sum(row["multiplicity"] for row in semantic_paths),
            "paths": semantic_paths,
            "query_id": query_id,
            "query_key": self._semantic_keys[query_id],
            "roles": roles,
            "source_ids": source_ids,
            "source_keys": sorted({self._semantic_keys[row_id] for row_id in source_ids}),
        }
        return {
            **payload,
            "query_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        }
