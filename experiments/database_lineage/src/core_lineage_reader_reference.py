"""Frozen scan-based Snapshot reader used only for index equivalence evaluation."""

from __future__ import annotations

from generation_relation_core.predicate_registry import PredicateRegistry
from generation_relation_core.snapshots import (
    SnapshotValidation,
    ValidatedSnapshot,
    validate_snapshot,
)

from .core_lineage_reader import LineageResult


class LegacyScanCoreLineageReader:
    """Resolve the same direct facts through repeated linear scans."""

    def __init__(
        self,
        snapshot: ValidatedSnapshot,
        registry: PredicateRegistry,
        *,
        prevalidated: SnapshotValidation | None = None,
    ) -> None:
        token = prevalidated or validate_snapshot(snapshot, registry)
        if token.snapshot_id != snapshot.snapshot_id:
            raise ValueError("prevalidation token does not belong to this snapshot")
        self.snapshot = snapshot

    @property
    def tables(self):
        return self.snapshot.tables

    def _support_for_tuple(self, tuple_id: str) -> dict:
        return next(
            row
            for row in self.tables.perceptual_support_records
            if row["support_payload"]["tuple_identity"] == tuple_id
        )

    def _source(self, source_id: str) -> dict:
        return next(
            row
            for row in self.tables.source_information_records
            if row["source_information_id"] == source_id
        )

    def _generated(self, generated_id: str) -> dict:
        return next(
            row
            for row in self.tables.generated_origins
            if row["generated_origin_id"] == generated_id
        )

    def _bindings_for_support(self, support_id: str) -> list[dict]:
        return [
            row
            for row in self.tables.generation_bindings
            if row["outcome_reference"].get("support_id") == support_id
        ]

    def _bindings_for_origin(self, kind: str, origin_id: str) -> list[dict]:
        return [
            row
            for row in self.tables.generation_bindings
            if row["origin_reference"]["kind"] == kind
            and row["origin_reference"].get(
                "source_information_id",
                row["origin_reference"].get("generated_origin_id"),
            )
            == origin_id
        ]

    def _generated_for_prior_support(self, support_id: str) -> list[dict]:
        return [
            row
            for row in self.tables.generated_origins
            if row["origin_payload"].get("bridge_kind") == "support_to_generated_origin"
            and row["origin_payload"].get("prior_support_id") == support_id
        ]

    def direct_input_tuple_ids(self, output_tuple_id: str) -> tuple[str, ...]:
        support_id = self._support_for_tuple(output_tuple_id)["support_id"]
        values = []
        for binding in self._bindings_for_support(support_id):
            origin = binding["origin_reference"]
            if origin["kind"] == "registered_source":
                values.append(
                    self._source(origin["source_information_id"])["source_identity"]
                )
            else:
                values.append(
                    self._generated(origin["generated_origin_id"])["origin_payload"][
                        "tuple_identity"
                    ]
                )
        return tuple(sorted(values))

    def backward(self, output_tuple_id: str) -> LineageResult:
        support_id = self._support_for_tuple(output_tuple_id)["support_id"]

        def walk(
            current_support: str, visited: frozenset[str]
        ) -> list[tuple[str, tuple[str, ...]]]:
            if current_support in visited:
                raise ValueError(f"cycle at support {current_support}")
            result = []
            for binding in sorted(
                self._bindings_for_support(current_support),
                key=lambda row: row["generation_binding_id"],
            ):
                binding_id = binding["generation_binding_id"]
                origin = binding["origin_reference"]
                if origin["kind"] == "registered_source":
                    result.append(
                        (
                            self._source(origin["source_information_id"])[
                                "source_identity"
                            ],
                            (binding_id,),
                        )
                    )
                    continue
                prior = self._generated(origin["generated_origin_id"])[
                    "origin_payload"
                ]["prior_support_id"]
                for identity, path in walk(prior, visited | {current_support}):
                    result.append((identity, (*path, binding_id)))
            return result

        paths = walk(support_id, frozenset())
        return LineageResult(
            tuple_ids=tuple(sorted({identity for identity, _path in paths})),
            derivation_path_count=len(paths),
            binding_paths=tuple(sorted(path for _identity, path in paths)),
        )

    def forward(
        self, source_tuple_id: str, final_output_tuple_ids: set[str]
    ) -> LineageResult:
        source = next(
            row
            for row in self.tables.source_information_records
            if row["source_identity"] == source_tuple_id
        )
        terminal_supports = {
            self._support_for_tuple(tuple_id)["support_id"]: tuple_id
            for tuple_id in final_output_tuple_ids
        }
        found = []

        def walk_origin(
            kind: str, origin_id: str, path: tuple[str, ...], visited: frozenset[str]
        ) -> None:
            key = f"{kind}:{origin_id}"
            if key in visited:
                raise ValueError(f"cycle at origin {key}")
            for binding in sorted(
                self._bindings_for_origin(kind, origin_id),
                key=lambda row: row["generation_binding_id"],
            ):
                outcome = binding["outcome_reference"]
                next_path = (*path, binding["generation_binding_id"])
                if outcome["kind"] != "support":
                    continue
                support_id = outcome["support_id"]
                if support_id in terminal_supports:
                    found.append((terminal_supports[support_id], next_path))
                for generated in self._generated_for_prior_support(support_id):
                    walk_origin(
                        "generated_origin",
                        generated["generated_origin_id"],
                        next_path,
                        visited | {key},
                    )

        walk_origin(
            "registered_source", source["source_information_id"], (), frozenset()
        )
        return LineageResult(
            tuple_ids=tuple(sorted({identity for identity, _path in found})),
            derivation_path_count=len(found),
            binding_paths=tuple(sorted(path for _identity, path in found)),
        )
