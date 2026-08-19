from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analysis import file_sha256, run_analysis
from .independent import check


DEFAULT_REPORT_ROOT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "target_support_branch_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]["all_runs"]
    diagnostic = result["pair_diagnostics"]
    decision = result["decision"]
    lines = [
        "# 目标级支撑—竞争边界分支实验",
        "",
        f"**裁决：`{decision['verdict']}`。**",
        "",
        "本轮没有重新训练或重跑GPU探针。实验从既有72个真实状态中读取更新前 alpha=0 的12种门控结果，按具体评价目标重建支撑结构与竞争边界；alpha>0结果未进入任何预测输入。",
        "",
        "## 目标级坐标是否区分组级剩余冲突",
        "",
        f"- 冻结的组级剩余案例：{diagnostic['remainder_count']}。",
        f"- 目标支撑距离超过开发普通对95%阈值：{diagnostic['remainder_above_target_support_threshold']}。",
        f"- 竞争边界距离超过该阈值：{diagnostic['remainder_above_competitor_boundary_threshold']}。",
        f"- 两块联合距离超过该阈值：{diagnostic['remainder_above_combined_threshold']}。",
        "",
        "## 同一64邻居池上的响应运输",
        "",
        "| 方法 | 全部曲线RMSE | 严重冲突RMSE | 剩余311例RMSE | 剩余311例方向准确率 | 剩余311例边界准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("f1_f3_f5", "target_support", "competitor_boundary", "target_support_competitor"):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['overall']['curve_rmse']:.5f} | {value['severe_conflict']['curve_rmse']:.5f} | "
            f"{value['group_level_remainder_311']['curve_rmse']:.5f} | "
            f"{value['group_level_remainder_311']['endpoint_direction_accuracy']:.4f} | "
            f"{value['group_level_remainder_311']['boundary_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"按剩余311例曲线RMSE选择的最佳方法为 `{decision['best_target_method_by_remainder_curve_rmse']}`；"
            f"相对F1/F3/F5改善 {decision['remainder_curve_rmse_relative_improvement']:.2%}，"
            f"边界准确率变化 {decision['remainder_boundary_accuracy_delta']:+.4f}。",
            "",
            "目标级坐标来自更新前真实门控事实。它们可以作为历史KNN的合法前态坐标；本报告没有把更新后的目标支撑或真实响应曲线塞回查询。",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_root: Path = DEFAULT_REPORT_ROOT) -> dict[str, Any]:
    if report_root.exists():
        raise RuntimeError(f"REPORT_ROOT_EXISTS:{report_root}")
    report_root.mkdir(parents=True)
    protocol = Path(__file__).with_name("PROTOCOL_FREEZE.md")
    shutil.copy2(protocol, report_root / "PROTOCOL_FREEZE.md")
    freeze = {
        "schema": "nanogpt-target-support-branch-experiment-freeze-v1",
        "status": "FROZEN_BEFORE_RESULTS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": file_sha256(protocol),
        "new_training": False,
        "gpu_execution": False,
        "new_probe": False,
        "allowed_alpha_values_for_executable_input": [0.0],
    }
    write_json(report_root / "EXPERIMENT_FREEZE.json", freeze)
    result = run_analysis()
    write_json(report_root / "FEATURE_MANIFEST.json", result["feature_manifest"])
    write_json(
        report_root / "TARGET_SUPPORT_RESPONSE_RESULTS.json",
        {"schema": "nanogpt-target-support-response-results-v1", "status": "PASS", "metrics": result["metrics"]},
    )
    write_json(report_root / "TARGET_PAIR_DIAGNOSTICS.json", result["pair_diagnostics"])
    write_json(report_root / "DECISION.json", result["decision"])
    write_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz", result["coordinates"]["source_ledger"])
    write_rows(report_root / "TARGET_COORDINATE_IDENTITY_LEDGER.jsonl.gz", result["coordinates"]["coordinate_rows"])
    write_rows(report_root / "PREDICTION_LEDGER.jsonl.gz", result["prediction_ledger"])
    write_rows(report_root / "TARGET_PAIR_DIAGNOSTIC_LEDGER.jsonl.gz", result["pair_ledger"])
    np.savez_compressed(
        report_root / "TARGET_COORDINATES.npz",
        record_ids=np.asarray([row["record_id"] for row in result["response"]["records"]]),
        target_support=result["coordinates"]["support"],
        competitor_boundary=result["coordinates"]["boundary"],
    )
    (report_root / "REPORT.md").write_text(_report(result), encoding="utf-8", newline="\n")
    independent = check(report_root)
    outputs = [path for path in report_root.iterdir() if path.is_file()]
    final = {
        "schema": "nanogpt-target-support-branch-final-manifest-v1",
        "status": "PASS",
        "scientific_status": result["decision"]["verdict"],
        "independent_check_status": independent["status"],
        "output_hashes": {path.name: file_sha256(path) for path in sorted(outputs)},
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
