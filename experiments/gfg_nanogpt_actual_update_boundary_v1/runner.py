from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .analysis import METHODS, PRIMARY, run_analysis


DEFAULT_OUTPUT = Path(r"E:\gfg-evidence\nanogpt-actual-update-boundary-v1\submission")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Actual-update target-boundary prediction",
        "",
        f"**Verdict: `{result['verdict']}`.**",
        "",
        "The experiment predicts only the immediate post-update target boundary. It does not predict response curves or CSRG support states and does not define a difficult-example subset.",
        "",
    ]
    for split in ("development", "confirmation", "all_runs"):
        lines.extend([
            f"## {split}",
            "",
            "| Method | Correct | Accuracy | Balanced accuracy | Four-way macro recall | Net repair vs linear |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            metric = result["metrics"][split][method]
            repair = result["repairs"][split][method]
            lines.append(
                f"| {method} | {metric['correct_count']}/{metric['count']} | "
                f"{metric['accuracy']:.6f} | {metric['balanced_accuracy']:.6f} | "
                f"{metric['four_way_macro_recall']:.6f} | {repair['net_repairs']:+d} |"
            )
        lines.append("")
    primary = result["metrics"]["confirmation"][PRIMARY]
    lines.extend([
        "## Primary result",
        "",
        f"The frozen primary algorithm predicted {primary['correct_count']} of {primary['count']} confirmation targets correctly "
        f"(accuracy {primary['accuracy']:.6%}; four-way macro recall {primary['four_way_macro_recall']:.6%}).",
        "",
        "All targets are retained under one rule. The result is a raw-state re-execution of an already established algorithm, not a new prospective confirmation.",
        "",
    ])
    return "\n".join(lines)


def run(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"OUTPUT_ROOT_EXISTS:{output_root}")
    output_root.mkdir(parents=True)
    protocol = Path(__file__).with_name("PROTOCOL_FREEZE.md")
    shutil.copy2(protocol, output_root / "PROTOCOL_FREEZE.md")
    write_json(output_root / "EXPERIMENT_CONTEXT.json", {
        "schema": "gfg-nanogpt-actual-update-boundary-context-v1",
        "status": "FROZEN_BEFORE_REEXECUTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": file_sha256(protocol),
        "evidence_status": "REEXECUTION_OF_ESTABLISHED_ALGORITHM_NOT_FRESH_CONFIRMATION",
        "primary_method": PRIMARY,
        "response_curve_predicted": False,
        "support_state_predicted": False,
        "difficult_subset_used": False,
    })
    result = run_analysis()
    write_json(output_root / "BOUNDARY_RESULTS.json", {
        key: value for key, value in result.items()
        if key not in ("ledger", "coordinates", "response_audit", "response_sources")
    })
    write_json(output_root / "DERIVATIVE_AUDIT.json", {
        "schema": "gfg-nanogpt-actual-update-boundary-derivative-audit-v1",
        "status": "PASS",
        "sections": result["coordinates"]["section_audits"],
    })
    write_json(output_root / "SOURCE_MANIFEST.json", {
        "schema": "gfg-nanogpt-actual-update-boundary-sources-v1",
        "response_sources": result["response_sources"],
        "response_audit": result["response_audit"],
    })
    write_ledger(output_root / "BOUNDARY_PREDICTIONS.jsonl.gz", result["ledger"])
    (output_root / "SCIENTIFIC_ASSESSMENT.md").write_text(_report(result), encoding="utf-8", newline="\n")
    names = (
        "PROTOCOL_FREEZE.md",
        "EXPERIMENT_CONTEXT.json",
        "BOUNDARY_RESULTS.json",
        "DERIVATIVE_AUDIT.json",
        "SOURCE_MANIFEST.json",
        "BOUNDARY_PREDICTIONS.jsonl.gz",
        "SCIENTIFIC_ASSESSMENT.md",
    )
    write_json(output_root / "MANIFEST.json", {
        "schema": "gfg-nanogpt-actual-update-boundary-manifest-v1",
        "files": {name: file_sha256(output_root / name) for name in names},
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output_root.resolve())
    primary = result["metrics"]["confirmation"][PRIMARY]
    print(json.dumps({
        "verdict": result["verdict"],
        "confirmation_accuracy": primary["accuracy"],
        "confirmation_four_way_macro_recall": primary["four_way_macro_recall"],
        "output_root": str(args.output_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

