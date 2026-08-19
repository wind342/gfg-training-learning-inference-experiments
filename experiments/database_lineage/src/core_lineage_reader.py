from __future__ import annotations

from dataclasses import dataclass

from generation_relation_core.predicate_registry import PredicateRegistry
from generation_relation_core.snapshots import (
    SnapshotValidation,
    ValidatedSnapshot,
    validate_snapshot,
)


@dataclass(frozen=True)
class LineageResult:
    tuple_ids: tuple[str, ...]
    derivation_path_count: int
    binding_paths: tuple[tuple[str, ...], ...]


class CoreLineageReader:
    """Traverse only validated Core entities, bindings, and GeneratedOrigin bridges."""

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
        tables = snapshot.tables
        self.snapshot = snapshot
        self.sources = {
            row["source_information_id"]: row
            for row in tables.source_information_records
        }
        self.source_id_by_tuple = {
            row["source_identity"]: row["source_information_id"]
            for row in self.sources.values()
        }
        self.generated = {
            row["generated_origin_id"]: row for row in tables.generated_origins
        }
        self.supports = {
            row["support_id"]: row for row in tables.perceptual_support_records
        }
        self.dispositions = {
            row["disposition_id"]: row for row in tables.explicit_dispositions
        }
        self.bindings = list(tables.generation_bindings)
        self.bindings_by_support: dict[str, list[dict]] = {}
        self.bindings_by_origin: dict[tuple[str, str], list[dict]] = {}
        for binding in self.bindings:
            outcome = binding["outcome_reference"]
            if outcome["kind"] == "support":
                self.bindings_by_support.setdefault(outcome["support_id"], []).append(
                    binding
                )
            origin = binding["origin_reference"]
            origin_id = origin.get(
                "source_information_id", origin.get("generated_origin_id")
            )
            self.bindings_by_origin.setdefault((origin["kind"], origin_id), []).append(
                binding
            )
        self.generated_by_prior_support: dict[str, list[dict]] = {}
        for row in self.generated.values():
            payload = row["origin_payload"]
            if (
                isinstance(payload, dict)
                and payload.get("bridge_kind") == "support_to_generated_origin"
            ):
                self.generated_by_prior_support.setdefault(
                    payload["prior_support_id"], []
                ).append(row)
        self.support_id_by_tuple = {
            row["support_payload"]["tuple_identity"]: row["support_id"]
            for row in self.supports.values()
        }

    @staticmethod
    def _origin_key(origin: dict) -> tuple[str, str]:
        return (
            origin["kind"],
            origin.get("source_information_id", origin.get("generated_origin_id")),
        )

    def direct_input_tuple_ids(self, output_tuple_id: str) -> tuple[str, ...]:
        support_id = self.support_id_by_tuple[output_tuple_id]
        values = []
        for binding in self.bindings_by_support.get(support_id, []):
            origin = binding["origin_reference"]
            if origin["kind"] == "registered_source":
                values.append(
                    self.sources[origin["source_information_id"]]["source_identity"]
                )
            else:
                values.append(
                    self.generated[origin["generated_origin_id"]]["origin_payload"][
                        "tuple_identity"
                    ]
                )
        return tuple(sorted(values))

    def backward(self, output_tuple_id: str) -> LineageResult:
        support_id = self.support_id_by_tuple[output_tuple_id]

        def walk(
            current_support: str, visited: frozenset[str]
        ) -> list[tuple[str, tuple[str, ...]]]:
            if current_support in visited:
                raise ValueError(f"cycle at support {current_support}")
            result: list[tuple[str, tuple[str, ...]]] = []
            for binding in sorted(
                self.bindings_by_support.get(current_support, []),
                key=lambda row: row["generation_binding_id"],
            ):
                binding_id = binding["generation_binding_id"]
                origin = binding["origin_reference"]
                if origin["kind"] == "registered_source":
                    identity = self.sources[origin["source_information_id"]][
                        "source_identity"
                    ]
                    result.append((identity, (binding_id,)))
                    continue
                generated = self.generated[origin["generated_origin_id"]]
                prior_support = generated["origin_payload"]["prior_support_id"]
                for identity, path in walk(prior_support, visited | {current_support}):
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
        source = self.sources[self.source_id_by_tuple[source_tuple_id]]
        terminal_supports = {
            self.support_id_by_tuple[item]: item for item in final_output_tuple_ids
        }
        found: list[tuple[str, tuple[str, ...]]] = []

        def walk_origin(
            kind: str, origin_id: str, path: tuple[str, ...], visited: frozenset[str]
        ) -> None:
            key = f"{kind}:{origin_id}"
            if key in visited:
                raise ValueError(f"cycle at origin {key}")
            for binding in sorted(
                self.bindings_by_origin.get((kind, origin_id), []),
                key=lambda row: row["generation_binding_id"],
            ):
                outcome = binding["outcome_reference"]
                next_path = (*path, binding["generation_binding_id"])
                if outcome["kind"] != "support":
                    continue
                support_id = outcome["support_id"]
                if support_id in terminal_supports:
                    found.append((terminal_supports[support_id], next_path))
                for generated in self.generated_by_prior_support.get(support_id, []):
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

    def direct_relations(self) -> list[dict[str, str]]:
        result = []
        for binding in sorted(
            self.bindings, key=lambda row: row["generation_binding_id"]
        ):
            origin = binding["origin_reference"]
            if origin["kind"] == "registered_source":
                input_tuple = self.sources[origin["source_information_id"]][
                    "source_identity"
                ]
            else:
                input_tuple = self.generated[origin["generated_origin_id"]][
                    "origin_payload"
                ]["tuple_identity"]
            outcome = binding["outcome_reference"]
            if outcome["kind"] == "support":
                output_tuple = self.supports[outcome["support_id"]]["support_payload"][
                    "tuple_identity"
                ]
                outcome_kind = "support"
            else:
                output_tuple = self.dispositions[outcome["disposition_id"]][
                    "disposition_payload"
                ]["tuple_identity"]
                outcome_kind = "disposition"
            result.append(
                {
                    "input_tuple_id": input_tuple,
                    "output_tuple_id": output_tuple,
                    "outcome_kind": outcome_kind,
                    "role": binding["relation_role"],
                    "binding_id": binding["generation_binding_id"],
                }
            )
        return result
