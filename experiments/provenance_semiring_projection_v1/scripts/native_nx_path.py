from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.native_nx import evaluate_native_nx
from experiments.provenance_semiring_projection_v1.src.workloads import load_workloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated Native N[X] K-relation path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.workload)
    results = []
    for workload in load_workloads():
        if selected and workload["id"] not in selected:
            continue
        variants = list(workload.get("queries", {"default": None}))
        for variant in variants:
            requested_variant = None if variant == "default" else variant
            results.append(evaluate_native_nx(workload, variant=requested_variant))
    document = {
        "schema_version": "native-nx-corpus-v1",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
