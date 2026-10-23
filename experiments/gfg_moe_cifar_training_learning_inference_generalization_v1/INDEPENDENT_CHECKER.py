from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .aggregate import aggregate
from .numeric import sha256_file


def _manifest_errors(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "FINAL_MANIFEST.json"
    if not manifest_path.exists():
        return ["MISSING_FINAL_MANIFEST"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.exists():
            errors.append(f"MISSING_FILE:{relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"BYTE_MISMATCH:{relative}")
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"HASH_MISMATCH:{relative}")
    return errors


def check(root: Path) -> dict[str, Any]:
    errors = _manifest_errors(root)
    result_path = root / "GENERALIZATION_RESULTS.json"
    if not result_path.exists():
        errors.append("MISSING_GENERALIZATION_RESULTS")
        return {"schema": "gfg-moe-independent-check-v1", "status": "FAIL", "errors": errors}
    recorded = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed = aggregate(root)
    if recorded != recomputed:
        errors.append("AGGREGATE_RECOMPUTATION_MISMATCH")
    expected_seeds = {20260910, 20260911, 20260912}
    observed_seeds = set(recomputed.get("seeds", []))
    if observed_seeds != expected_seeds:
        errors.append("FORMAL_SEED_SET_MISMATCH")
    for seed in sorted(expected_seeds & observed_seeds):
        seed_root = root / f"seed_{seed}"
        gfg = json.loads((seed_root / "COMPACT_GFG.json").read_text(encoding="utf-8"))
        validation = gfg.get("validation", {})
        if validation.get("status") != "PASS" or validation.get("errors"):
            errors.append(f"GFG_VALIDATION_FAILED:{seed}")
        facts = gfg.get("facts", [])
        required = {"u", "tau", "omega", "z", "rho"}
        if any(not required.issubset(row) for row in facts):
            errors.append(f"INCOMPLETE_ATOMIC_FACT:{seed}")
        summary = json.loads((seed_root / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        if summary["integrity"]["persistent_model_state_frozen"] is not True:
            errors.append(f"MODEL_STATE_CHANGED_DURING_INFERENCE:{seed}")
        if summary["integrity"]["persistent_optimizer_state_frozen"] is not True:
            errors.append(f"OPTIMIZER_STATE_CHANGED_DURING_INFERENCE:{seed}")
    return {
        "schema": "gfg-moe-independent-check-v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "joint_verdict": recorded.get("joint_verdict"),
        "recomputed_joint_verdict": recomputed.get("joint_verdict"),
        "formal_seeds": sorted(observed_seeds),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
