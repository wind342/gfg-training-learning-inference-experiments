from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.native_lower_k import (
    evaluate_direct_lower_domains,
)
from experiments.provenance_semiring_projection_v1.src.workloads import load_workloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated direct lower-K evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for workload in load_workloads():
        for variant in workload.get("queries", {"default": None}):
            requested = None if variant == "default" else variant
            results.append(evaluate_direct_lower_domains(workload, variant=requested))
    document = {
        "schema_version": "native-direct-lower-k-corpus-v2",
        "process_role": "direct target carriers from base annotations and frozen RA AST",
        "computes_nx_first": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
