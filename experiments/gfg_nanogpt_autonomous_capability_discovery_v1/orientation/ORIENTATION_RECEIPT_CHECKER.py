from __future__ import annotations

import json
from pathlib import Path
import sys


REQUIRED_TEXT_FIELDS = (
    "generation_fact",
    "generation_occurrence",
    "atomic_generation_fact",
    "realizes_fact",
    "reads_from",
    "GeneratedOrigin",
    "program_order",
    "equal_values",
    "cartesian_recombination",
    "missing_relations",
    "forward_query",
    "reverse_query",
    "executable_claim",
    "state_sufficiency",
    "prefix_only_falsification",
    "report_code_correspondence",
    "operational_state_use",
    "intervention_state_audit",
    "cross_run_invariance",
    "dual_dynamics_decomposition",
    "full_horizon_stability",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python ORIENTATION_RECEIPT_CHECKER.py orientation_receipt.json")
        return 2
    path = Path(sys.argv[1])
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ORIENTATION_RECEIPT_INVALID_JSON:{type(exc).__name__}")
        return 1

    failures: list[str] = []
    if receipt.get("schema") != "gfg-orientation-receipt-v1":
        failures.append("schema")
    if receipt.get("read_complete") is not True:
        failures.append("read_complete")
    if receipt.get("target_gfg_accessed") is not False:
        failures.append("target_gfg_accessed")
    for field in REQUIRED_TEXT_FIELDS:
        value = receipt.get(field)
        if not isinstance(value, (str, list, dict)) or not value:
            failures.append(field)

    if failures:
        print("ORIENTATION_RECEIPT_MISSING_OR_INVALID:" + ",".join(failures))
        return 1
    print("ORIENTATION_RECEIPT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
