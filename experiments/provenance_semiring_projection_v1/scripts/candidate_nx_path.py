from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation_relation_core.predicate_registry import PredicateRegistry
from generation_relation_core.snapshots import CoreV3Tables, ValidatedSnapshot, validate_snapshot

from experiments.provenance_semiring_projection_v1.src.candidate_nx import project_snapshot_to_nx
from experiments.provenance_semiring_projection_v1.src.profile_runtime import support_membership_predicate


def _validated_snapshot(document: dict) -> tuple[ValidatedSnapshot, object]:
    tables = CoreV3Tables(**document["tables"])
    snapshot = ValidatedSnapshot(record=document["record"], tables=tables)
    registry = PredicateRegistry(
        tables.support_space_records,
        tables.predicate_profiles,
        {tables.predicate_profiles[0]["predicate_profile_id"]: support_membership_predicate},
    )
    validation = validate_snapshot(snapshot, registry)
    return snapshot, validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Core-only Candidate N[X] projection")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.input.read_text(encoding="utf-8"))
    if corpus.get("schema_version") != "core-snapshot-corpus-v1":
        raise ValueError("unexpected Core snapshot corpus")
    results = []
    for item in corpus["results"]:
        snapshot, validation = _validated_snapshot(item["snapshot"])
        projection = project_snapshot_to_nx(snapshot, validation)
        results.append(
            {
                "workload_id": item["workload_id"],
                "variant": item["variant"],
                "snapshot_id": snapshot.snapshot_id,
                "source_variables": projection["source_variables"],
                "outputs": projection["outputs"],
            }
        )
    output = {"schema_version": "core-projected-nx-corpus-v1", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
