from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PACKAGE_PREFIX = "experiments.pytorch_autograd_training_lineage_v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parent


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _dependencies() -> list[dict[str, str]]:
    rows = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(PACKAGE_PREFIX):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        except ValueError:
            continue
        rows.append({
            "module": name,
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def _write(path: str, value: dict[str, Any]) -> None:
    Path(path).write_bytes(_canonical(value))


def _candidate(snapshot_input: str, output: str) -> None:
    from .gradient_dependency_comparison import extract_core_gradient_dependencies

    snapshots = json.loads(Path(snapshot_input).read_text(encoding="utf-8"))
    relations = extract_core_gradient_dependencies(snapshots)
    _write(output, {
        "audit": {
            "candidate_oracle_read_count": 0,
            "candidate_receipt_reference_read_count": 0,
            "candidate_saved_tensor_artifact_read_count": 0,
            "candidate_validated_snapshot_read_count": 1,
            "role": "candidate_core_normalizer",
        },
        "dependencies": _dependencies(),
        "relations": relations,
    })


def _native(output: str) -> None:
    from .gradient_intervention_oracle import run_gradient_intervention_oracle

    result = run_gradient_intervention_oracle()
    _write(output, {
        "audit": {
            "native_candidate_read_count": 0,
            "native_core_binding_read_count": 0,
            "native_core_read_count": 0,
            "native_old_reference_read_count": 0,
            "role": "native_pytorch_oracle",
        },
        "checkpoint_recomputation_replay_equivalence": result[
            "checkpoint_recomputation_replay_equivalence"
        ],
        "dependencies": _dependencies(),
        "relations": result["native_gradient_dependency_oracle"]["relations"],
    })


def _comparison(core_input: str, native_input: str, output: str) -> None:
    from .gradient_dependency_comparison import compare_gradient_dependencies

    core = json.loads(Path(core_input).read_text(encoding="utf-8"))
    native = json.loads(Path(native_input).read_text(encoding="utf-8"))
    comparison = compare_gradient_dependencies(
        core["relations"],
        native["relations"],
        checkpoint_replay_equivalence=native[
            "checkpoint_recomputation_replay_equivalence"
        ],
    )
    _write(output, {
        "audit": {
            "comparison_input_count": 2,
            "comparison_mutation_count": 0,
            "role": "read_only_relation_comparison",
        },
        "comparison": comparison,
        "dependencies": _dependencies(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate", "native", "comparison"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-input")
    parser.add_argument("--core-input")
    parser.add_argument("--native-input")
    args = parser.parse_args()
    if args.mode == "candidate":
        if args.snapshot_input is None:
            raise ValueError("CANDIDATE_SNAPSHOT_INPUT_REQUIRED")
        _candidate(args.snapshot_input, args.output)
    elif args.mode == "native":
        _native(args.output)
    else:
        if args.core_input is None or args.native_input is None:
            raise ValueError("COMPARISON_INPUTS_REQUIRED")
        _comparison(args.core_input, args.native_input, args.output)


if __name__ == "__main__":
    main()
