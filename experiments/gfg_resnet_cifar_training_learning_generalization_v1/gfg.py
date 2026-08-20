from __future__ import annotations

from typing import Any

from .numeric import canonical_json_bytes, sha256_bytes


def _identity(prefix: str, payload: Any) -> str:
    return f"{prefix}_{sha256_bytes(canonical_json_bytes(payload))}"


class CompactGFG:
    """A compact fact graph over registered scientific occurrences only."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.sources: dict[str, dict[str, Any]] = {}
        self.occurrences: dict[str, dict[str, Any]] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.facts: list[dict[str, Any]] = []

    def source(self, kind: str, payload: dict[str, Any]) -> str:
        row = {"kind": kind, "payload": payload}
        identity = _identity("src", row)
        self.sources[identity] = {"source_id": identity, **row}
        return identity

    def occurrence(self, kind: str, payload: dict[str, Any]) -> str:
        row = {"kind": kind, "payload": payload}
        identity = _identity("occ", row)
        self.occurrences[identity] = {"occurrence_id": identity, **row}
        return identity

    def outcome(self, kind: str, payload: dict[str, Any]) -> str:
        row = {"kind": kind, "payload": payload}
        identity = _identity("out", row)
        self.outcomes[identity] = {"outcome_id": identity, **row}
        return identity

    def fact(
        self,
        source_id: str,
        transformation: str,
        occurrence_id: str,
        outcome_id: str,
        role: str,
    ) -> str:
        row = {
            "u": source_id,
            "tau": transformation,
            "omega": occurrence_id,
            "z": outcome_id,
            "rho": role,
        }
        identity = _identity("fact", row)
        self.facts.append({"fact_id": identity, **row})
        return identity

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        fact_ids: set[str] = set()
        for fact in self.facts:
            if fact["fact_id"] in fact_ids:
                errors.append(f"DUPLICATE_FACT:{fact['fact_id']}")
            fact_ids.add(fact["fact_id"])
            if fact["u"] not in self.sources:
                errors.append(f"MISSING_SOURCE:{fact['u']}")
            if fact["omega"] not in self.occurrences:
                errors.append(f"MISSING_OCCURRENCE:{fact['omega']}")
            if fact["z"] not in self.outcomes:
                errors.append(f"MISSING_OUTCOME:{fact['z']}")
            if not fact["tau"] or not fact["rho"]:
                errors.append(f"EMPTY_RELATION:{fact['fact_id']}")
        return {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "source_count": len(self.sources),
            "occurrence_count": len(self.occurrences),
            "outcome_count": len(self.outcomes),
            "fact_count": len(self.facts),
        }

    def document(self) -> dict[str, Any]:
        validation = self.validate()
        document = {
            "schema": "compact-registered-scientific-gfg-v1",
            "run_id": self.run_id,
            "sources": sorted(self.sources.values(), key=lambda row: row["source_id"]),
            "occurrences": sorted(
                self.occurrences.values(), key=lambda row: row["occurrence_id"]
            ),
            "outcomes": sorted(
                self.outcomes.values(), key=lambda row: row["outcome_id"]
            ),
            "facts": sorted(self.facts, key=lambda row: row["fact_id"]),
            "validation": validation,
        }
        document["graph_sha256"] = sha256_bytes(canonical_json_bytes(document))
        return document


def add_update_event(
    graph: CompactGFG,
    event_index: int,
    epoch: int,
    batch_ids: list[int],
    training_loss: float,
    analysis: dict[str, Any],
) -> None:
    receiving = graph.source(
        "parameter_sgd_receiving_state",
        {"sha256": analysis["pre_state_sha256"], "epoch": epoch},
    )
    batch = graph.source(
        "identified_cifar100_training_batch", {"sample_ids": batch_ids}
    )
    update = graph.outcome(
        "formed_actual_state_update",
        {"sha256": analysis["delta_state_sha256"], "loss": training_loss},
    )
    update_occurrence = graph.occurrence(
        "actual_sgd_training_update", {"event_index": event_index, "epoch": epoch}
    )
    graph.fact(receiving, "sgd_momentum_training_step", update_occurrence, update, "receiving_state")
    graph.fact(batch, "sgd_momentum_training_step", update_occurrence, update, "training_source")
    response = graph.outcome(
        "finite_amplitude_target_response",
        {
            "target_ids": analysis["selected_target_ids"],
            "alpha_grid": analysis["alpha_grid"],
        },
    )
    response_occurrence = graph.occurrence(
        "finite_amplitude_replay", {"event_index": event_index, "epoch": epoch}
    )
    graph.fact(receiving, "apply_realized_update_path", response_occurrence, response, "receiving_state")
    update_source = graph.source(
        "generated_actual_update", {"outcome_id": update, "sha256": analysis["delta_state_sha256"]}
    )
    graph.fact(update_source, "apply_realized_update_path", response_occurrence, response, "update_action")
    support = graph.outcome(
        "distributed_support_reallocation",
        {"target_ids": analysis["selected_target_ids"], "component_count": 4},
    )
    support_occurrence = graph.occurrence(
        "residual_stage_coalition_gating",
        {"event_index": event_index, "coalition_count_per_state": 16},
    )
    response_source = graph.source(
        "generated_target_response", {"outcome_id": response}
    )
    graph.fact(response_source, "support_coalition_adjudication", support_occurrence, support, "functional_response")
