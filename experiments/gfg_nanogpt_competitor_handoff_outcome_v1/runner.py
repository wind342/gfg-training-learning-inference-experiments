from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analysis import METHODS, file_sha256, run_analysis
from .independent import check


DEFAULT_REPORT_ROOT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "competitor_handoff_outcome_v1"


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


def _table(metrics: dict[str, Any], repairs: dict[str, Any], split: str) -> list[str]:
    lines = [
        f"### {split}",
        "",
        "| 方法 | 结果正确数 | 准确率 | 平衡准确率 | 修复 | 破坏 | 净修复 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = metrics[split][method]["group_level_remainder_311"]
        repair = repairs[split][method]["group_level_remainder_311"]
        lines.append(
            f"| {method} | {value['correct_count']}/{value['count']} | {value['accuracy']:.4f} | "
            f"{value['balanced_accuracy']:.4f} | {repair['fixed_baseline_errors']} | "
            f"{repair['newly_broken_baseline_answers']} | {repair['net_repairs']:+d} |"
        )
    return lines


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# 多竞争者换手—最终结果实验",
        "",
        f"**裁决：`{result['decision']['verdict']}`。**",
        "",
        "本轮只预测更新后的最终正确/错误结果，不以响应曲线拟合选择模型。全部竞争者坐标由更新前完整logits、当前输出参数和已经形成的实际更新计算；任何alpha>0结果均未进入输入。",
        "",
        "## 冻结311例的最终结果",
        "",
    ]
    lines.extend(_table(result["metrics"], result["repairs"], "all_runs"))
    lines.append("")
    lines.extend(_table(result["metrics"], result["repairs"], "confirmation"))
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "`all_competitor_gaps`只增加当前23个竞争者的排序差距；`all_competitor_geometry`增加当前分类边界与实际更新的配准；联合模型同时使用两者。输出行空间直接作用只是合法的更新前估计，不被描述为完整功能JVP。",
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
    write_json(
        report_root / "EXPERIMENT_FREEZE.json",
        {
            "schema": "nanogpt-competitor-handoff-outcome-freeze-v1",
            "status": "FROZEN_BEFORE_RESULTS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": file_sha256(protocol),
            "new_training": False,
            "gpu_execution": False,
            "new_probe": False,
            "primary_target": "final_correct_or_wrong",
            "curve_metric_is_primary": False,
        },
    )
    result = run_analysis()
    write_json(report_root / "FEATURE_MANIFEST.json", result["feature_manifest"])
    write_json(report_root / "OUTCOME_RESULTS.json", {"schema": "nanogpt-competitor-handoff-outcome-results-v1", "status": "PASS", "metrics": result["metrics"]})
    write_json(report_root / "OUTCOME_REPAIRS.json", {"schema": "nanogpt-competitor-handoff-repairs-v1", "status": "PASS", "repairs": result["repairs"]})
    write_json(report_root / "DECISION.json", result["decision"])
    write_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz", result["coordinates"]["source_rows"])
    write_rows(report_root / "COORDINATE_IDENTITY_LEDGER.jsonl.gz", result["coordinates"]["coordinate_rows"])
    write_rows(report_root / "OUTCOME_PREDICTION_LEDGER.jsonl.gz", result["prediction_ledger"])
    np.savez_compressed(
        report_root / "COMPETITOR_COORDINATES.npz",
        record_ids=np.asarray([row["record_id"] for row in result["response"]["records"]]),
        all_competitor_gaps=result["coordinates"]["gaps"],
        all_competitor_geometry=result["coordinates"]["geometry"],
    )
    (report_root / "REPORT.md").write_text(_report(result), encoding="utf-8", newline="\n")
    independent = check(report_root)
    outputs = [path for path in report_root.iterdir() if path.is_file()]
    final = {
        "schema": "nanogpt-competitor-handoff-outcome-final-v1",
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
