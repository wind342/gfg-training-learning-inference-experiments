from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analysis import DEFAULT_TRAINER_ROOT, METHODS, run_analysis
from .independent import check


DEFAULT_REPORT_ROOT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "full_network_receptive_state_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# 全网络目标微观功能接收态诊断",
        "",
        f"**裁决：`{result['decision']['verdict']}`。**",
        "",
        "本实验是事后机制诊断。全网络JVP只用于检验缺失接收态假设，不被登记为正式在线预测输入。",
        "",
    ]
    for split in ("development", "confirmation", "all_runs"):
        lines.extend([f"## {split}", "", "| 方法 | 全部正确 | 全部净修复 | 311例正确 | 311例净修复 |", "|---|---:|---:|---:|---:|"])
        for method in METHODS:
            overall = result["metrics"][split][method]["overall"]
            hard = result["metrics"][split][method]["group_level_remainder_311"]
            overall_repair = result["repairs"][split][method]["overall"]
            hard_repair = result["repairs"][split][method]["group_level_remainder_311"]
            lines.append(
                f"| {method} | {overall['correct_count']}/{overall['count']} | {overall_repair['net_repairs']:+d} | "
                f"{hard['correct_count']}/{hard['count']} | {hard_repair['net_repairs']:+d} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(report_root: Path, trainer_root: Path) -> dict[str, Any]:
    if report_root.exists():
        raise RuntimeError(f"REPORT_ROOT_EXISTS:{report_root}")
    report_root.mkdir(parents=True)
    protocol = Path(__file__).with_name("PROTOCOL_FREEZE.md")
    shutil.copy2(protocol, report_root / "PROTOCOL_FREEZE.md")
    _json(
        report_root / "EXPERIMENT_CONTEXT.json",
        {
            "schema": "nanogpt-full-network-receptive-state-context-v1",
            "status": "FROZEN_BEFORE_JVP_RESULTS_AFTER_PRIOR_OUTCOMES_EXPOSED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": _sha(protocol),
            "evidence_status": "POST_HOC_MECHANISM_DIAGNOSTIC_ONLY",
            "new_training": False,
            "gpu_execution": False,
            "new_response_probe": True,
            "new_gfg": False,
        },
    )
    result = run_analysis(trainer_root)
    _json(report_root / "FEATURE_MANIFEST.json", result["feature_manifest"])
    _json(report_root / "SECTION_JVP_AUDIT.json", {"schema": "nanogpt-full-network-section-jvp-audit-v1", "status": "PASS", "sections": result["receptive"]["section_audits"]})
    _json(report_root / "OUTCOME_RESULTS.json", {"schema": "nanogpt-full-network-receptive-state-results-v1", "status": "PASS", "metrics": result["metrics"], "runwise": result["runwise"]})
    _json(report_root / "OUTCOME_REPAIRS.json", {"schema": "nanogpt-full-network-receptive-state-repairs-v1", "status": "PASS", "repairs": result["repairs"]})
    _json(report_root / "DECISION.json", result["decision"])
    _rows(report_root / "OUTCOME_PREDICTION_LEDGER.jsonl.gz", result["prediction_ledger"])
    _rows(report_root / "COORDINATE_IDENTITY_LEDGER.jsonl.gz", result["receptive"]["coordinate_rows"])
    np.savez_compressed(
        report_root / "RECEPTIVE_COORDINATES.npz",
        record_ids=np.asarray([row["record_id"] for row in result["response"]["records"]]),
        total_gap_jvp=result["receptive"]["total_gap_jvp"],
        component_gap_jvp=result["receptive"]["component_gap_jvp"],
    )
    np.savez_compressed(report_root / "RETRIEVAL_NEIGHBORS.npz", **result["neighbors"])
    (report_root / "REPORT.md").write_text(_report(result), encoding="utf-8", newline="\n")
    independent = check(report_root)
    outputs = sorted(path for path in report_root.iterdir() if path.is_file())
    final = {
        "schema": "nanogpt-full-network-receptive-state-final-v1",
        "status": "PASS",
        "scientific_status": result["decision"]["verdict"],
        "evidence_status": result["decision"]["evidence_status"],
        "independent_check_status": independent["status"],
        "output_hashes": {path.name: _sha(path) for path in outputs},
    }
    _json(report_root / "FINAL_MANIFEST.json", final)
    (report_root / "READY").write_text("READY\n", encoding="utf-8", newline="\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--trainer-root", type=Path, default=DEFAULT_TRAINER_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.report_root.resolve(), args.trainer_root.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
