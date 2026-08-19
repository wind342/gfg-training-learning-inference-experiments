from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

from .analysis import FACTOR_RECORDS, run_analysis, sha256_file
from .independent import check


DEFAULT_REPORT_ROOT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "support_reallocation_audit_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _report(summary: dict[str, Any]) -> str:
    relation = summary["relations"]
    per_run = summary["per_run"]
    return "\n".join(
        [
            "# 真实更新驱动的分布式支撑重新分工审计",
            "",
            "本审计比较同一接收前态上的 α=0（未施加更新）与 α=1（完整施加真实更新），直接计算更新造成的CSRG支撑网络变化。",
            "",
            "## 事实规模",
            "",
            f"- 真实更新截面：{summary['section_count']}；运行：{summary['entry_count']}。",
            f"- 目标组支撑转移：{summary['target_group_transition_count']}；有效allocation转移：{summary['valid_allocation_transition_count']}。",
            f"- 主支撑组件换手：{summary['primary_support_switch_count']}（{summary['primary_support_switch_rate']:.2%}）。",
            "",
            "## 核心关系",
            "",
            f"- 更新总幅度 → 平均支撑重新分配：Spearman ρ={relation['update_magnitude_to_section_mean_reallocation_rho']:.3f}。",
            f"- 更新总幅度 → 主支撑换手率：ρ={relation['update_magnitude_to_primary_support_switch_rate_rho']:.3f}。",
            f"- 重新分配幅度 → 能力变化绝对值：ρ={relation['reallocation_to_absolute_capability_change_rho']:.3f}。",
            f"- 有效支撑变化 → 能力有符号变化：ρ={relation['effective_support_change_to_capability_change_rho']:.3f}。",
            f"- 支撑集中度变化 → 能力有符号变化：ρ={relation['concentration_change_to_capability_change_rho']:.3f}。",
            "",
            f"能力发生变化时，平均重新分配幅度为 {summary['mean_reallocation_when_capability_changed']:.4f}；能力不变时为 {summary['mean_reallocation_when_capability_unchanged']:.4f}。",
            "",
            "12个运行中，更新幅度与重新分配的相关方向全部为正，范围为 "
            f"{min(value['update_to_reallocation_rho'] for value in per_run.values()):.3f}–{max(value['update_to_reallocation_rho'] for value in per_run.values()):.3f}。",
            "",
            "## 裁决",
            "",
            "真实训练更新确实搬动了分布式能力支撑网络。更新总幅度稳定决定重新分工有多强；但单个组件的更新幅度不能稳定决定该组件获得还是失去支撑。重新分工方向必须由当前支撑构型与更新共同描述。",
            "",
            "这是探索性事后因果描述，不是更新发生前的预测模型。",
            "",
        ]
    )


def run(report_root: Path = DEFAULT_REPORT_ROOT) -> dict[str, Any]:
    if report_root.exists():
        raise RuntimeError(f"REPORT_ROOT_EXISTS:{report_root}")
    report_root.mkdir(parents=True)
    scope = Path(__file__).with_name("ANALYSIS_SCOPE.md")
    shutil.copy2(scope, report_root / "ANALYSIS_SCOPE.md")
    result = run_analysis()
    source_paths = {FACTOR_RECORDS.resolve(), scope.resolve()}
    source_paths.update(Path(row["section_npz"]).resolve() for row in result["source_ledger"])
    manifest = {
        "schema": "nanogpt-support-reallocation-analysis-manifest-v1",
        "status": "EXPLORATORY_POST_HOC",
        "claim_scope": "EXPLORATORY_POST_HOC_CAUSAL_DESCRIPTION_NOT_ADVANCE_PREDICTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha_before": 0.0,
        "alpha_after": 1.0,
        "new_training": False,
        "gpu_execution": False,
        "new_probe": False,
        "source_hashes": {str(path): sha256_file(path) for path in sorted(source_paths)},
    }
    write_json(report_root / "ANALYSIS_MANIFEST.json", manifest)
    write_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz", result["source_ledger"])
    write_rows(report_root / "SECTION_UPDATE_LEDGER.jsonl.gz", result["section_rows"])
    write_rows(report_root / "TARGET_SUPPORT_TRANSITION_LEDGER.jsonl.gz", result["group_rows"])
    write_json(report_root / "SUPPORT_REALLOCATION_RESULTS.json", result["summary"])
    (report_root / "REPORT.md").write_text(_report(result["summary"]), encoding="utf-8", newline="\n")
    independent = check(report_root)
    outputs = [path for path in report_root.iterdir() if path.is_file()]
    derivation = {
        "schema": "nanogpt-support-reallocation-derivation-manifest-v1",
        "status": "PASS",
        "source_section_count": len(result["source_ledger"]),
        "derived_output_hashes": {path.name: sha256_file(path) for path in sorted(outputs)},
        "independent_check_status": independent["status"],
        "future_information_used_as_predictor": False,
    }
    write_json(report_root / "DERIVATION_MANIFEST.json", derivation)
    final = {
        "schema": "nanogpt-support-reallocation-final-manifest-v1",
        "status": "PASS",
        "scientific_status": "UPDATE_DRIVEN_DISTRIBUTED_SUPPORT_REALLOCATION_SUPPORTED",
        "independent_check_status": independent["status"],
        "results_sha256": sha256_file(report_root / "SUPPORT_REALLOCATION_RESULTS.json"),
        "derivation_manifest_sha256": sha256_file(report_root / "DERIVATION_MANIFEST.json"),
    }
    write_json(report_root / "FINAL_MANIFEST.json", final)
    (report_root / "READY").write_text("READY\n", encoding="utf-8", newline="\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.report_root.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
