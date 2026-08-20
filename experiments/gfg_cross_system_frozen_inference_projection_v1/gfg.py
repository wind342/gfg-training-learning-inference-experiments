from __future__ import annotations

from typing import Any


class ProjectionGFG:
    def __init__(self) -> None:
        self.occurrences: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []

    def add(
        self,
        *,
        occurrence_id: str,
        kind: str,
        source: str,
        transformation: str,
        outcome: str,
        role: str,
        evidence: dict[str, Any],
    ) -> None:
        fact_id = f"fact:{occurrence_id}:{len(self.facts)}"
        self.occurrences.append(
            {"id": occurrence_id, "kind": kind, "realizes_fact": fact_id}
        )
        self.facts.append(
            {
                "id": fact_id,
                "u": source,
                "tau": transformation,
                "omega": occurrence_id,
                "z": outcome,
                "rho": role,
                "evidence": evidence,
            }
        )

    def document(self) -> dict[str, Any]:
        occurrence_ids = [row["id"] for row in self.occurrences]
        fact_ids = [row["id"] for row in self.facts]
        errors: list[str] = []
        if len(occurrence_ids) != len(set(occurrence_ids)):
            errors.append("DUPLICATE_OCCURRENCE_ID")
        if len(fact_ids) != len(set(fact_ids)):
            errors.append("DUPLICATE_FACT_ID")
        by_occurrence = {row["omega"] for row in self.facts}
        if set(occurrence_ids) != by_occurrence:
            errors.append("OCCURRENCE_FACT_INCIDENCE_MISMATCH")
        required = {"u", "tau", "omega", "z", "rho"}
        if any(not required.issubset(row) for row in self.facts):
            errors.append("INCOMPLETE_ATOMIC_FACT")
        return {
            "schema": "gfg-cross-system-frozen-inference-projection-v1",
            "occurrences": self.occurrences,
            "facts": self.facts,
            "validation": {
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
                "occurrence_count": len(self.occurrences),
                "fact_count": len(self.facts),
            },
        }
