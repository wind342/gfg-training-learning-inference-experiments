from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .core_snapshot import build_core_snapshot
from .graph_artifacts import validate_graph, write_artifacts
from .run_experiment import STEPS, TRAINER_COMMIT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "artifacts"
            / "runtime_receipts_checkpoint.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    args = parser.parse_args()
    capture = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    snapshot, graph = build_core_snapshot(
        capture,
        trainer_commit=TRAINER_COMMIT,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        device_name=torch.cuda.get_device_name(0),
        source_code_path=(
            Path(__file__).resolve().parent / "run_experiment.py"
        ),
    )
    parameter_count = sum(
        1
        for row in capture["sources"]
        if row["source_payload"].get("source_kind")
        == "initialized_model_parameter"
    )
    validation = validate_graph(
        capture,
        graph,
        expected_steps=STEPS,
        parameter_count=parameter_count,
    )
    if validation["status"] != "PASS":
        raise RuntimeError(
            "GENERATION_FACT_GRAPH_VALIDATION_FAILED:"
            + json.dumps(validation, sort_keys=True)
        )
    paths = write_artifacts(
        args.output_dir,
        capture,
        snapshot,
        graph,
        validation,
        expected_steps=STEPS,
    )
    manifest = {
        "artifact_files": [path.name for path in paths],
        "capture_checkpoint": args.checkpoint.name,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "nanoGPT_commit": TRAINER_COMMIT,
        "torch": torch.__version__,
        "validation": validation,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
