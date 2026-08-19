from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from experiments.database_lineage.src.metrics import write_json
from generation_relation_core.canonical import finalize_entity
from generation_relation_core.implementation_identity import IMPLEMENTATION_FILES


RUNTIME = Path("experiments/database_lineage/runtime")
BASELINE = RUNTIME / "determinism_baseline"
ARTIFACT = Path("experiments/database_lineage/artifacts/tpch_determinism.json")
FIRST_RUN_SUMMARY = Path("experiments/database_lineage/artifacts/tpch_first_run.json")
LEGACY_FIRST_RUN_CORE_COMMIT = "50a21fd4dfc7ca7cd1e939e844c2ca07877e8a16"


def _implementation_hashes_at(commit: str) -> dict[str, str]:
    hashes = {}
    for relative, _role in IMPLEMENTATION_FILES:
        data = subprocess.check_output(["git", "show", f"{commit}:{relative}"])
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def _same_snapshot_content_across_core_identity(
    first_snapshot: dict,
    second_snapshot_record: dict,
    first_implementation_hashes: dict[str, str],
) -> bool:
    # Snapshot IDs intentionally bind the tracked Core implementation.  The
    # resolver changed between these executions, so reconstruct the second
    # run's envelope under the first run's implementation identity.  Equality
    # then proves that every authoritative/derived table count and hash, plus
    # every manifest/operation/environment ID, stayed byte-identical.
    if "snapshot_record" in first_snapshot:
        first_payload = {
            key: value
            for key, value in first_snapshot["snapshot_record"].items()
            if key
            not in {"snapshot_id", "snapshot_payload_sha256", "implementation_hashes"}
        }
        second_payload = {
            key: value
            for key, value in second_snapshot_record.items()
            if key
            not in {"snapshot_id", "snapshot_payload_sha256", "implementation_hashes"}
        }
        return first_payload == second_payload

    payload = {
        key: value
        for key, value in second_snapshot_record.items()
        if key not in {"snapshot_id", "snapshot_payload_sha256"}
    }
    payload["implementation_hashes"] = first_implementation_hashes
    reconstructed = finalize_entity("ValidatedSnapshot", payload)
    return reconstructed["snapshot_id"] == first_snapshot["snapshot_id"]


def main() -> int:
    queries = {}
    all_equal = True
    first_run_summary = json.loads(FIRST_RUN_SUMMARY.read_text(encoding="utf-8"))
    first_run_core_commit = first_run_summary.get(
        "core_commit",
        LEGACY_FIRST_RUN_CORE_COMMIT,
    )
    first_implementation_hashes = _implementation_hashes_at(first_run_core_commit)
    used_legacy_snapshot_normalization = False
    for query in (1, 3, 6, 10):
        name = f"tpch_result_sf_0_01_q{query}.json"
        lineage_name = f"core_lineage_sf_0_01_q{query}.json"
        before = json.loads((BASELINE / name).read_text(encoding="utf-8"))
        used_legacy_snapshot_normalization |= (
            "snapshot_record" not in before["snapshot"]
        )
        after = json.loads((RUNTIME / name).read_text(encoding="utf-8"))
        before_lineage = (BASELINE / lineage_name).read_bytes()
        after_lineage = (RUNTIME / lineage_name).read_bytes()
        before_backward = json.loads(before_lineage)
        expected_forward: dict[str, list[str]] = {}
        for output_key, sources in before_backward.items():
            for source in sources:
                expected_forward.setdefault(source, []).append(output_key)
        for outputs in expected_forward.values():
            outputs.sort()
        after_forward = json.loads(
            (RUNTIME / f"core_forward_lineage_sf_0_01_q{query}.json").read_text(
                encoding="utf-8"
            )
        )
        forward_equal = all(
            sorted(outputs) == expected_forward.get(source, [])
            for source, outputs in after_forward.items()
        ) and set(expected_forward) <= set(after_forward)
        checks = {
            "raw_snapshot_id_equal": before["snapshot"]["snapshot_id"]
            == after["snapshot"]["snapshot_id"],
            "normalized_snapshot_content_equal": _same_snapshot_content_across_core_identity(
                before["snapshot"],
                after["snapshot"]["snapshot_record"],
                first_implementation_hashes,
            ),
            "entity_counts_equal": before["snapshot"]["entity_counts"]
            == after["snapshot"]["entity_counts"],
            "binding_count_equal": before["snapshot"]["binding_count"]
            == after["snapshot"]["binding_count"],
            "csv_hash_equal": before["output"]["csv_sha256"]
            == after["output"]["csv_sha256"],
            "json_hash_equal": before["output"]["json_sha256"]
            == after["output"]["json_sha256"],
            "backward_path_counts_equal": before["lineage"]["path_counts"]
            == after["lineage"]["path_counts"],
            "backward_lineage_bytes_equal": before_lineage == after_lineage,
            "forward_lineage_equal_to_inverted_first_run_backward": forward_equal,
        }
        exact = all(
            value for name, value in checks.items() if name != "raw_snapshot_id_equal"
        )
        all_equal = all_equal and exact
        queries[f"q{query}"] = {
            "exact": exact,
            "checks": checks,
            "before_lineage_sha256": hashlib.sha256(before_lineage).hexdigest(),
            "after_lineage_sha256": hashlib.sha256(after_lineage).hexdigest(),
            "after_forward_lineage_sha256": hashlib.sha256(
                (RUNTIME / f"core_forward_lineage_sf_0_01_q{query}.json").read_bytes()
            ).hexdigest(),
        }
    write_json(
        ARTIFACT,
        {
            "scale_factor": 0.01,
            "first_run_core_commit": first_run_core_commit,
            "normalization": (
                "Snapshot IDs bind implementation hashes. Full Snapshot envelopes are compared "
                "after removing only implementation-derived fields. For the retained legacy "
                "baseline without an envelope, the second envelope is re-finalized with the "
                "first run's tracked Core hashes. Both paths cover all authoritative and derived "
                "table counts and hashes while allowing an audited implementation identity change."
            ),
            "legacy_snapshot_id_reconstruction_used": used_legacy_snapshot_normalization,
            "queries": queries,
            "all_equal": all_equal,
        },
    )
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
