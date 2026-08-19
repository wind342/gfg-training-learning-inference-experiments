from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analysis import (
    ALPHAS,
    BASE_SEED,
    BOOTSTRAP_REPLICATES,
    GLOBAL_UNSEEN_ENTRY,
    PRIMARY_CALIPER_QUANTILE,
    PRIMARY_K,
    RANDOM_PERMUTATIONS,
    _block_distance_squared,
    build_feature_space,
    build_records,
    canonical_json,
    compare_configurations,
    file_sha256,
    match_configuration,
    read_json,
    require,
    sha256_bytes,
    summarize_configuration,
    write_json,
    write_match_ledger,
)


BLOCKS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
CURRENT_BLOCKS = ("F1", "F2", "F3", "F4", "F5")


def _nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, float) and not np.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.extend(_nonfinite_paths(child, f"{prefix}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            result.extend(_nonfinite_paths(child, f"{prefix}[{index}]"))
        return result
    return []


def _write_records(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in records:
            invalid = _nonfinite_paths(row)
            require(not invalid, f"NONFINITE_RECORD_VALUE:{row.get('record_id')}:{invalid[:8]}")
            handle.write(canonical_json(row) + "\n")
    return {"path": path.name, "sha256": file_sha256(path), "row_count": len(records), "compression": "gzip"}


def _read_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decision(comparison: dict[str, Any]) -> str:
    shape = comparison["improvements"]["normalized_shape_correlation"]
    secondary_names = (
        "normalized_shape_rmse",
        "endpoint_abs_difference",
        "response_type_agreement",
        "boundary_class_agreement",
        "competitor_switch_agreement",
    )
    secondaries = [comparison["improvements"][name] for name in secondary_names]
    if shape["estimate"] is None:
        return "NOT_EVALUABLE"
    if float(shape["ci95_low"]) > 0 and any(row["ci95_low"] is not None and float(row["ci95_low"]) > 0 for row in secondaries):
        return "SUPPORTED"
    if float(shape["estimate"]) > 0:
        return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def _random_distribution(records: list[dict[str, Any]], *, pool: str, subset: set[int], label: str) -> dict[str, Any]:
    by_key: dict[Any, list[int]] = {}
    for index in subset:
        key = records[index]["target_group"] if pool == "group" else records[index]["semantic_target_key"]
        by_key.setdefault(key, []).append(index)
    q = np.asarray([np.asarray(row["response"]["normalized_curve"], dtype=np.float64)[[2, 3, 4, 5, 6]] for row in records])
    q_center = q - np.mean(q, axis=1, keepdims=True)
    q_norm = np.linalg.norm(q_center, axis=1)
    draws: list[float] = []
    rng = np.random.default_rng(BASE_SEED + (0 if pool == "group" else 1))
    eligible = sorted(subset)
    candidate_lists: list[np.ndarray] = []
    query_indices: list[int] = []
    for index in eligible:
        key = records[index]["target_group"] if pool == "group" else records[index]["semantic_target_key"]
        candidates = [value for value in by_key[key] if records[value]["entry_id"] != records[index]["entry_id"]]
        if candidates:
            query_indices.append(index)
            candidate_lists.append(np.asarray(candidates, dtype=np.int64))
    query_array = np.asarray(query_indices, dtype=np.int64)
    for _ in range(RANDOM_PERMUTATIONS):
        references = np.asarray([values[int(rng.integers(0, len(values)))] for values in candidate_lists], dtype=np.int64)
        denom = q_norm[query_array] * q_norm[references]
        valid = denom > 1e-12
        correlation = np.sum(q_center[query_array] * q_center[references], axis=1)
        correlation = correlation[valid] / denom[valid]
        draws.append(float(np.mean(correlation)))
    return {
        "label": label,
        "pool": pool,
        "permutations": RANDOM_PERMUTATIONS,
        "eligible_query_count": len(query_indices),
        "mean": float(np.mean(draws)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "draws_sha256": sha256_bytes(np.asarray(draws, dtype=np.float64).tobytes()),
    }


def _local_extrapolation_unchanged(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    h = 0.125
    for record in records:
        margins = np.asarray(record["response"]["margin_curve"], dtype=np.float64)
        m0 = float(margins[1])
        derivative = float((margins[2] - margins[0]) / (2 * h))
        curvature = float((margins[2] - 2 * margins[1] + margins[0]) / (h * h))
        predicted_j = m0 + derivative
        predicted_jk = m0 + derivative + 0.5 * curvature
        truth = record["response"]["boundary_class"]
        if truth not in {"MAINTAIN_CORRECT", "MAINTAIN_WRONG"}:
            continue
        start = m0 >= 0
        j_cross = (predicted_j >= 0) != start
        jk_cross = (predicted_jk >= 0) != start
        rows.append(
            {
                "record_id": record["record_id"],
                "section_id": record["section_id"],
                "evaluation_unit_id": record["evaluation_unit_id"],
                "truth": truth,
                "response_type": record["response"]["response_type"],
                "competitor_switch": record["response"]["competitor_switch"],
                "j_false_crossing": bool(j_cross),
                "jk_false_crossing": bool(jk_cross),
                "endpoint_delta": record["response"]["endpoint_delta"],
                "nonlinear_residual_rms": record["response"]["response_type_detail"]["nonlinear_residual_rms"],
                "support_concentration": record["features"]["F4"]["numeric"]["support_concentration"],
                "effective_support": record["features"]["F4"]["numeric"]["effective_support"],
            }
        )
    def group_counts(key: str) -> dict[str, Any]:
        selected = [row for row in rows if row[key]]
        return {
            "count": len(selected),
            "response_type_counts": dict(Counter(row["response_type"] for row in selected)),
            "competitor_switch_count": sum(bool(row["competitor_switch"]) for row in selected),
            "mean_nonlinear_residual_rms": float(np.mean([row["nonlinear_residual_rms"] for row in selected])) if selected else None,
            "mean_support_concentration": float(np.mean([row["support_concentration"] for row in selected])) if selected else None,
            "mean_effective_support": float(np.mean([row["effective_support"] for row in selected])) if selected else None,
        }
    return {
        "schema": "nanogpt-unchanged-target-local-extrapolation-analysis-v1",
        "status": "PASS",
        "unchanged_target_count": len(rows),
        "j_false_crossing": group_counts("j_false_crossing"),
        "jk_false_crossing": group_counts("jk_false_crossing"),
        "rows": rows,
        "interpretation_boundary": "Associations with response type, switching and support are descriptive, not causal.",
    }


def _surviving_counterexamples(records: list[dict[str, Any]], space: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for match in full["rows"]:
        query = int(match["query_index"])
        reference = int(match["reference_indices"][0])
        metrics = match["metrics"]
        left = records[query]
        right = records[reference]
        shape_bad = metrics["normalized_shape_correlation"] is not None and float(metrics["normalized_shape_correlation"]) < 0
        direction_bad = np.sign(left["response"]["endpoint_delta"]) != np.sign(right["response"]["endpoint_delta"])
        boundary_bad = left["response"]["boundary_class"] != right["response"]["boundary_class"]
        if not (shape_bad or direction_bad or boundary_bad):
            continue
        block_distances = {
            block: float(np.sqrt(_block_distance_squared(space, block, query, np.asarray([reference]))[0]))
            for block in CURRENT_BLOCKS
        }
        largest = max(block_distances, key=block_distances.get)
        classification = {
            "F1": "缺少边界状态细节",
            "F2": "缺少竞争者结构",
            "F3": "缺少目标特异更新信息",
            "F4": "缺少支撑状态",
            "F5": "缺少优化器接收态",
        }[largest]
        rows.append(
            {
                "query_record_id": left["record_id"],
                "reference_record_id": right["record_id"],
                "query_identity": {key: left[key] for key in ("section_id", "evaluation_unit_id", "semantic_target_key", "target_group")},
                "reference_identity": {key: right[key] for key in ("section_id", "evaluation_unit_id", "semantic_target_key", "target_group")},
                "matching_distance": float(match["distances"][0]),
                "block_distances": block_distances,
                "response_metrics": metrics,
                "query_response": left["response"],
                "reference_response": right["response"],
                "provisional_classification": classification,
                "classification_is_not_causal": True,
                "source_refs": {"query": left["source_refs"], "reference": right["source_refs"]},
            }
        )
    rows.sort(
        key=lambda row: (
            row["response_metrics"]["normalized_shape_correlation"] if row["response_metrics"]["normalized_shape_correlation"] is not None else 1.0,
            -row["response_metrics"]["endpoint_abs_difference"],
        )
    )
    return {
        "schema": "nanogpt-surviving-conditional-response-counterexamples-v1",
        "status": "PASS",
        "definition": "full-current-condition match passes primary caliper but shape correlation <0, endpoint direction differs, or boundary class differs",
        "counterexample_count": len(rows),
        "rows": rows,
    }


def _assessment_markdown(
    block_decisions: dict[str, str],
    history_decisions: dict[str, str],
    identity_decision: str,
    survivors: int,
    full_summary: dict[str, Any],
) -> str:
    supported = [block for block, status in block_decisions.items() if status == "SUPPORTED"]
    partial = [block for block, status in block_decisions.items() if status == "PARTIALLY_SUPPORTED"]
    shape = full_summary["metrics"]["normalized_shape_correlation"]
    lines = [
        "# 科学评估",
        "",
        "## 结论",
        "",
        "本轮只识别非线性响应形成以前的统计条件，没有训练预测器，也没有访问全局未见运行。",
        "",
        f"完整当前条件匹配后的归一化形态相关性点估计为 `{shape['estimate']}`，95%运行对聚类区间为 `[{shape['ci95_low']}, {shape['ci95_high']}]`。",
        "",
        f"得到充分支持的条件块：{supported if supported else '无'}；部分支持：{partial if partial else '无'}。",
        "",
        "## 通俗解释",
        "",
        "- margin本身描述离边界多远，但不能单独表示竞争者换手、支撑重组和更新作用，因此只能解释响应的一部分。",
        "- 竞争者结构检验的是同样margin下，错误类别的排序和拥挤程度是否改变完整曲线。",
        "- 更新几何区分全局更新大小、组件分配以及正确类与当前竞争类输出行受到的真实更新；它不读取本步响应结果。",
        "- 支撑状态来自响应以前已经完成的CSRG诊断层，能够检验备用与接管是否对应饱和、回弯和最终不跨越，但它不是普通训练天然字段。",
        "- Adam块检验的是实际参数更新已经形成以后，接收态优化器结构是否仍携带独立统计信息；统计增益不能被表述为因果作用。",
        f"- 自然历史增益判决：`{history_decisions.get('F6', 'NOT_EVALUABLE')}`；过去响应曲线的额外诊断增益：`{history_decisions.get('F7', 'NOT_EVALUABLE')}`。",
        f"- 当前条件下仍保留 `{survivors}` 个幸存条件反例，说明任何正面平均结果都不能被写成完整运输律。",
        f"- 目标身份残余判决：`{identity_decision}`。若残余存在，它代表当前状态仍未表达的语义、支撑位置或目标特异更新关系，不能直接把sample_id当特征。",
        "",
        "## 当前最小候选状态",
        "",
        "当前只能把获得SUPPORTED或PARTIALLY_SUPPORTED的前置条件块列为候选状态组成；本轮没有证明这些块已经充分，也没有建立可执行响应函数。",
        "",
        "## 下一阶段边界",
        "",
        "只有在本轮条件块与反例均完成审计后，下一阶段才可以用这些前置条件建立无当前探针的响应预测函数，并在严格按运行留出的数据上验证。",
    ]
    return "\n".join(lines) + "\n"


def run_factor_analysis(
    *,
    response_root: Path,
    stepwise_root: Path,
    output_root: Path,
    contract_path: Path,
    original_task_path: Path,
    resume_frozen: bool = False,
    global_unseen_diagnostic_accessed: bool = False,
) -> dict[str, Any]:
    if resume_frozen:
        require(output_root.is_dir(), f"FROZEN_OUTPUT_MISSING:{output_root}")
        require((output_root / "ANALYSIS_FREEZE.json").is_file(), "ANALYSIS_FREEZE_MISSING")
        freeze = read_json(output_root / "ANALYSIS_FREEZE.json")
        require(freeze["status"] == "FROZEN_BEFORE_FACTOR_RESULTS", "ANALYSIS_FREEZE_STATUS_INVALID")
        require(file_sha256(output_root / "FACTOR_ANALYSIS_CONTRACT.md") == freeze["contract_sha256"], "FROZEN_CONTRACT_CHANGED")
        require(file_sha256(output_root / "ORIGINAL_TASK.txt") == freeze["original_task_sha256"], "FROZEN_TASK_CHANGED")
        completed_record_stage = False
        factor_schema_path = output_root / "FACTOR_SCHEMA.json"
        record_path = output_root / "PRETARGET_FACTOR_RECORDS.jsonl.gz"
        if factor_schema_path.is_file() and record_path.is_file():
            prior_factor_schema = read_json(factor_schema_path)
            record_manifest = prior_factor_schema.get("record_manifest", {})
            completed_record_stage = (
                prior_factor_schema.get("status") == "PASS"
                and int(record_manifest.get("row_count", -1)) == 15264
                and record_manifest.get("sha256") == file_sha256(record_path)
            )
        if not completed_record_stage:
            attempt_number = 1
            while (output_root / f"failed-attempt-{attempt_number:02d}").exists():
                attempt_number += 1
            archive = output_root / f"failed-attempt-{attempt_number:02d}"
            archive.mkdir(exist_ok=True)
            archived: list[dict[str, Any]] = []
            for name in ("PRETARGET_FEATURE_AVAILABILITY.json", "PRETARGET_FACTOR_RECORDS.jsonl.gz"):
                source = output_root / name
                if source.is_file():
                    destination = archive / name
                    require(not destination.exists(), f"RECOVERY_ARCHIVE_EXISTS:{destination}")
                    archived.append({"name": name, "sha256": file_sha256(source), "bytes": source.stat().st_size})
                    shutil.move(str(source), str(destination))
            prior_recovery = read_json(output_root / "RECOVERY_AUDIT.json") if (output_root / "RECOVERY_AUDIT.json").is_file() else None
            attempts = list(prior_recovery.get("attempts", [])) if prior_recovery else []
            attempts.append(
                {
                    "attempt": attempt_number,
                    "archive_directory": archive.name,
                    "reason": "analysis process terminated before a complete pretarget record stream was established",
                    "archived_incomplete_outputs": archived,
                }
            )
            write_json(
                output_root / "RECOVERY_AUDIT.json",
                {
                    "schema": "nanogpt-response-factor-analysis-recovery-v1",
                    "status": "RESUMED_FROM_FROZEN_CONTRACT",
                    "attempts": attempts,
                    "contract_or_source_result_changed": False,
                },
            )
    else:
        require(not output_root.exists(), f"OUTPUT_ALREADY_EXISTS:{output_root}")
        output_root.mkdir(parents=True)
        shutil.copy2(contract_path, output_root / "FACTOR_ANALYSIS_CONTRACT.md")
        shutil.copy2(original_task_path, output_root / "ORIGINAL_TASK.txt")
    source_names = (
        "SELECTION_MANIFEST.json",
        "IDENTITY_MATERIAL.json",
        "RESOLVED_INVENTORY.json",
        "UPDATE_GEOMETRY_CONTROL_MANIFEST.json",
        "FINITE_AMPLITUDE_CURVES_MANIFEST.json",
        "VALIDATION.json",
        "INDEPENDENT_REPLAY.json",
    )
    source_hashes = {name: file_sha256(response_root / name) for name in source_names}
    section_hashes = {
        path.name: file_sha256(path)
        for path in sorted((response_root / "sections").glob("*"))
        if path.is_file()
    }
    if resume_frozen:
        require(freeze["source_hashes"] == source_hashes, "FROZEN_SOURCE_HASHES_CHANGED")
        require(int(freeze["section_file_count"]) == len(section_hashes), "FROZEN_SECTION_COUNT_CHANGED")
        require(freeze["section_hash_set_sha256"] == sha256_bytes(canonical_json(section_hashes).encode("utf-8")), "FROZEN_SECTION_HASHES_CHANGED")
    else:
        freeze = {
            "schema": "nanogpt-response-factor-analysis-freeze-v1",
            "status": "FROZEN_BEFORE_FACTOR_RESULTS",
            "contract_sha256": file_sha256(output_root / "FACTOR_ANALYSIS_CONTRACT.md"),
            "original_task_sha256": file_sha256(output_root / "ORIGINAL_TASK.txt"),
            "source_hashes": source_hashes,
            "section_file_count": len(section_hashes),
            "section_hash_set_sha256": sha256_bytes(canonical_json(section_hashes).encode("utf-8")),
            "alpha_grid": [float(value) for value in ALPHAS],
            "global_unseen_entry": GLOBAL_UNSEEN_ENTRY,
            "global_unseen_entry_accessed": False,
            "prediction_model_training_authorized": False,
        }
        write_json(output_root / "ANALYSIS_FREEZE.json", freeze)
    if global_unseen_diagnostic_accessed:
        write_json(
            output_root / "BOUNDARY_VIOLATION.json",
            {
                "schema": "nanogpt-response-factor-analysis-boundary-violation-v1",
                "status": "GLOBAL_UNSEEN_DIAGNOSTIC_ACCESS_DISCLOSED",
                "entry_id": GLOBAL_UNSEEN_ENTRY,
                "scope": "one source-GFG optimizer-step object-role/schema diagnostic",
                "finite_amplitude_response_accessed": False,
                "used_in_factor_analysis": False,
                "strict_unseen_claim_remains_valid": False,
                "required_remedy_for_future_hidden_adjudication": "reserve a new execution never accessed by this analysis session",
            },
        )

    if resume_frozen and completed_record_stage:
        records = _read_records(output_root / "PRETARGET_FACTOR_RECORDS.jsonl.gz")
        require(len(records) == 15264, "COMPLETED_RECORD_STAGE_ROW_COUNT_INVALID")
        availability = read_json(output_root / "PRETARGET_FEATURE_AVAILABILITY.json")
        record_manifest = read_json(output_root / "FACTOR_SCHEMA.json")["record_manifest"]
    else:
        records, availability = build_records(response_root, stepwise_root)
        write_json(output_root / "PRETARGET_FEATURE_AVAILABILITY.json", availability)
        record_manifest = _write_records(output_root / "PRETARGET_FACTOR_RECORDS.jsonl.gz", records)
    space, standardization = build_feature_space(records)
    factor_schema = {
        "schema": "nanogpt-response-factor-schema-v1",
        "status": "PASS",
        "blocks": standardization["blocks"],
        "record_manifest": record_manifest,
        "identity_fields": ["evaluation_unit_id", "row_content_sha256", "upstream_element_identity", "target_group"],
        "cross_run_semantic_key": ["row_content_sha256", "target_group"],
        "array_position_is_identity": False,
        "global_unseen_entry_accessed": bool(global_unseen_diagnostic_accessed),
    }
    write_json(output_root / "FACTOR_SCHEMA.json", factor_schema)
    matching_protocol = {
        "schema": "nanogpt-response-factor-matching-protocol-v1",
        "status": "FROZEN_BEFORE_MATCH_RESULTS",
        "primary": {"k": PRIMARY_K, "caliper_quantile": PRIMARY_CALIPER_QUANTILE, "cross_run": True},
        "sensitivity": {"k": [1, 3, 5], "caliper_quantile": [0.75, 0.90, 0.95, 1.0]},
        "standardization": standardization,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_permutations": RANDOM_PERMUTATIONS,
        "seed": BASE_SEED,
        "outcome_not_used_for_matching": True,
    }
    write_json(output_root / "MATCHING_PROTOCOL.json", matching_protocol)

    all_indices = set(range(len(records)))
    f7_indices = {
        index
        for index, record in enumerate(records)
        if record["features"]["F7"]["categorical"].get("availability") == "AVAILABLE"
    }
    require(len(f7_indices) == 36 * 212, "F7_SUBSET_COUNT_INVALID")

    primary_configs: list[dict[str, Any]] = []
    c0 = match_configuration(records, space, name="C0_random_cross_run", blocks=(), pool="group", random_selection=True)
    c1 = match_configuration(records, space, name="C1_semantic_identity", blocks=(), pool="semantic", random_selection=True)
    primary_configs.extend((c0, c1))
    cumulative: dict[str, dict[str, Any]] = {"C0": c0, "C1": c1}
    current_blocks: list[str] = []
    for number, block in enumerate(CURRENT_BLOCKS, start=2):
        current_blocks.append(block)
        config = match_configuration(
            records,
            space,
            name=f"C{number}_{'_'.join(current_blocks)}",
            blocks=tuple(current_blocks),
            pool="semantic",
        )
        cumulative[f"C{number}"] = config
        primary_configs.append(config)
    c7 = match_configuration(records, space, name="C7_current_plus_native_history", blocks=CURRENT_BLOCKS + ("F6",), pool="semantic")
    cumulative["C7"] = c7
    primary_configs.append(c7)
    c7_sub = match_configuration(
        records,
        space,
        name="C7_on_prior_curve_subset",
        blocks=CURRENT_BLOCKS + ("F6",),
        pool="semantic",
        subset=f7_indices,
    )
    c8 = match_configuration(
        records,
        space,
        name="C8_current_native_history_prior_curve",
        blocks=CURRENT_BLOCKS + ("F6", "F7"),
        pool="semantic",
        subset=f7_indices,
    )
    primary_configs.extend((c7_sub, c8))

    single_configs: dict[str, dict[str, Any]] = {}
    for block in BLOCKS:
        subset = f7_indices if block == "F7" else all_indices
        config = match_configuration(records, space, name=f"single_{block}", blocks=(block,), pool="semantic", subset=subset)
        single_configs[block] = config
        primary_configs.append(config)

    full_current = cumulative["C6"]
    lofo_configs: dict[str, dict[str, Any]] = {}
    for block in CURRENT_BLOCKS:
        kept = tuple(value for value in CURRENT_BLOCKS if value != block)
        config = match_configuration(records, space, name=f"current_without_{block}", blocks=kept, pool="semantic")
        lofo_configs[block] = config
        primary_configs.append(config)

    current_sub = match_configuration(records, space, name="current_on_prior_curve_subset", blocks=CURRENT_BLOCKS, pool="semantic", subset=f7_indices)
    current_history_sub = c7_sub
    current_history_curve_sub = c8
    primary_configs.append(current_sub)

    incremental_summaries = {name: summarize_configuration(config, name) for name, config in cumulative.items()}
    incremental_comparisons: dict[str, Any] = {}
    ordered = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
        incremental_comparisons[f"{previous}_to_{current}"] = compare_configurations(cumulative[current], cumulative[previous], f"{previous}_to_{current}")
    incremental_comparisons["C7_to_C8_same_subset"] = compare_configurations(c8, c7_sub, "C7_to_C8_same_subset")
    incremental_results = {
        "schema": "nanogpt-response-incremental-conditioning-v1",
        "status": "PASS",
        "summaries": incremental_summaries,
        "increments": incremental_comparisons,
        "C8_summary": summarize_configuration(c8, "C8"),
        "C7_same_subset_summary": summarize_configuration(c7_sub, "C7_subset"),
    }
    write_json(output_root / "INCREMENTAL_CONDITIONING_RESULTS.json", incremental_results)

    single_results = {
        "schema": "nanogpt-response-single-factor-results-v1",
        "status": "PASS",
        "random_group_baseline": summarize_configuration(c0, "C0_single"),
        "semantic_identity_baseline": summarize_configuration(c1, "C1_single"),
        "random_permutation_distributions": {
            "group": _random_distribution(records, pool="group", subset=all_indices, label="group"),
            "semantic": _random_distribution(records, pool="semantic", subset=all_indices, label="semantic"),
        },
        "factors": {},
    }
    for block, config in single_configs.items():
        baseline = c1
        if block == "F7":
            baseline = match_configuration(records, space, name="C1_semantic_identity_F7_subset", blocks=(), pool="semantic", subset=f7_indices, random_selection=True)
        comparison = compare_configurations(config, baseline, f"single_{block}")
        single_results["factors"][block] = {
            "summary": summarize_configuration(config, f"single_{block}"),
            "increment_over_semantic_identity": comparison,
            "decision": _decision(comparison),
        }
    write_json(output_root / "SINGLE_FACTOR_RESULTS.json", single_results)

    lofo_results = {
        "schema": "nanogpt-response-leave-one-factor-out-v1",
        "status": "PASS",
        "full_current_summary": summarize_configuration(full_current, "full_current"),
        "factors": {},
    }
    block_decisions: dict[str, str] = {}
    for block, config in lofo_configs.items():
        comparison = compare_configurations(full_current, config, f"restore_{block}")
        decision = _decision(comparison)
        block_decisions[block] = decision
        lofo_results["factors"][block] = {
            "without_summary": summarize_configuration(config, f"without_{block}"),
            "restoration_increment": comparison,
            "decision": decision,
        }
    write_json(output_root / "LEAVE_ONE_FACTOR_OUT_RESULTS.json", lofo_results)

    history_native = compare_configurations(current_history_sub, current_sub, "history_F6_increment")
    history_curve = compare_configurations(current_history_curve_sub, current_history_sub, "history_F7_increment")
    history_decisions = {"F6": _decision(history_native), "F7": _decision(history_curve)}
    history_results = {
        "schema": "nanogpt-response-history-increment-v1",
        "status": "PASS",
        "same_subset_record_count": len(f7_indices),
        "current": summarize_configuration(current_sub, "history_current"),
        "current_plus_native_history": summarize_configuration(current_history_sub, "history_native"),
        "current_plus_native_history_plus_prior_curve": summarize_configuration(current_history_curve_sub, "history_curve"),
        "native_history_increment": history_native,
        "prior_curve_increment": history_curve,
        "decisions": history_decisions,
        "prior_curve_role": "PRIOR_ACTIVE_DIAGNOSTIC_MEMORY_NOT_CURRENT_STEP_INPUT",
    }
    write_json(output_root / "HISTORY_INCREMENT_RESULTS.json", history_results)

    response_type_results = {
        "schema": "nanogpt-response-type-conditioning-v1",
        "status": "PASS",
        "factor_agreement_metrics": {
            block: {
                key: value["summary"]["metrics"][key]
                for key in (
                    "response_type_agreement",
                    "competitor_switch_agreement",
                    "first_switch_alpha_abs_difference",
                    "first_zero_alpha_abs_difference",
                    "boundary_class_agreement",
                    "correct_to_wrong_agreement",
                    "wrong_to_correct_agreement",
                    "maintain_correct_agreement",
                    "maintain_wrong_agreement",
                )
            }
            for block, value in single_results["factors"].items()
        },
    }
    write_json(output_root / "RESPONSE_TYPE_CONDITIONING.json", response_type_results)

    unchanged = _local_extrapolation_unchanged(records)
    write_json(output_root / "UNCHANGED_TARGET_ANALYSIS.json", unchanged)

    identity_different = match_configuration(
        records,
        space,
        name="full_current_different_identity",
        blocks=CURRENT_BLOCKS,
        pool="group",
        exclude_same_semantic=True,
    )
    identity_comparison = compare_configurations(full_current, identity_different, "same_vs_different_identity")
    identity_decision = "IDENTITY_RESIDUAL_REMAINS" if _decision(identity_comparison) in {"SUPPORTED", "PARTIALLY_SUPPORTED"} else "IDENTITY_RESIDUAL_NOT_ESTABLISHED"
    identity_results = {
        "schema": "nanogpt-response-identity-residual-v1",
        "status": "PASS",
        "same_semantic_identity": summarize_configuration(full_current, "identity_same"),
        "different_identity_same_group": summarize_configuration(identity_different, "identity_different"),
        "same_over_different_increment": identity_comparison,
        "decision": identity_decision,
        "sample_id_used_as_feature": False,
    }
    write_json(output_root / "IDENTITY_RESIDUAL_RESULTS.json", identity_results)

    survivors = _surviving_counterexamples(records, space, full_current)
    write_json(output_root / "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json", survivors)

    alternative_orders = (
        ("F3", "F1", "F4", "F2", "F5"),
        ("F4", "F2", "F1", "F5", "F3"),
        ("F5", "F3", "F2", "F4", "F1"),
    )
    alternative_results: dict[str, Any] = {}
    for order_index, order in enumerate(alternative_orders, start=1):
        previous = c1
        prefixes: dict[str, Any] = {}
        active: list[str] = []
        for block in order:
            active.append(block)
            config = match_configuration(records, space, name=f"alt{order_index}_{'_'.join(active)}", blocks=tuple(active), pool="semantic")
            prefixes[block] = {
                "summary": summarize_configuration(config, f"alt{order_index}_{block}"),
                "increment": compare_configurations(config, previous, f"alt{order_index}_{block}"),
            }
            previous = config
        alternative_results[f"order_{order_index}"] = {"order": list(order), "prefixes": prefixes}

    sensitivity: dict[str, Any] = {"full_current": {}, "block_restoration": {block: {} for block in CURRENT_BLOCKS}}
    for quantile in (0.75, 0.90, 0.95, 1.0):
        full = match_configuration(records, space, name=f"sensitivity_full_q{quantile}", blocks=CURRENT_BLOCKS, pool="semantic", caliper_quantile=quantile)
        sensitivity["full_current"][f"k1_q{quantile}"] = summarize_configuration(full, f"sensitivity_full_q{quantile}")
        for block in CURRENT_BLOCKS:
            without = match_configuration(
                records,
                space,
                name=f"sensitivity_without_{block}_q{quantile}",
                blocks=tuple(value for value in CURRENT_BLOCKS if value != block),
                pool="semantic",
                caliper_quantile=quantile,
            )
            sensitivity["block_restoration"][block][f"k1_q{quantile}"] = compare_configurations(full, without, f"sensitivity_{block}_q{quantile}")
    for k in (3, 5):
        full = match_configuration(records, space, name=f"sensitivity_full_k{k}", blocks=CURRENT_BLOCKS, pool="semantic", k=k)
        sensitivity["full_current"][f"k{k}_q0.9"] = summarize_configuration(full, f"sensitivity_full_k{k}")

    robustness = {
        "schema": "nanogpt-response-factor-robustness-v1",
        "status": "PASS",
        "alternative_orders": alternative_results,
        "matching_sensitivity": sensitivity,
        "statistical_unit": "unordered run pair; section/update pair sensitivity retained in match ledger",
        "targets_are_not_independent_replicates": True,
    }
    write_json(output_root / "ROBUSTNESS_AND_SENSITIVITY.json", robustness)

    match_manifest = write_match_ledger(output_root / "MATCH_LEDGER.jsonl.gz", primary_configs + [identity_different])
    write_json(output_root / "MATCH_LEDGER_MANIFEST.json", match_manifest)
    assessment = _assessment_markdown(
        block_decisions,
        history_decisions,
        identity_decision,
        survivors["counterexample_count"],
        lofo_results["full_current_summary"],
    )
    (output_root / "SCIENTIFIC_ASSESSMENT.md").write_text(assessment, encoding="utf-8", newline="\n")

    manifest = {
        "schema": "nanogpt-response-factor-analysis-manifest-v1",
        "status": "ANALYSIS_COMPLETE_PENDING_GFG_VALIDATION_WITH_DISCLOSED_BOUNDARY_VIOLATION" if global_unseen_diagnostic_accessed else "ANALYSIS_COMPLETE_PENDING_GFG_VALIDATION",
        "record_count": len(records),
        "section_count": 72,
        "entry_count": 12,
        "prior_curve_subset_record_count": len(f7_indices),
        "block_decisions": block_decisions,
        "history_decisions": history_decisions,
        "identity_decision": identity_decision,
        "surviving_counterexample_count": survivors["counterexample_count"],
        "match_ledger": match_manifest,
        "global_unseen_entry_accessed": bool(global_unseen_diagnostic_accessed),
        "prediction_model_trained": False,
        "deliverables": {},
    }
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["deliverables"][path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    write_json(output_root / "MANIFEST.json", manifest)
    return manifest


__all__ = ["run_factor_analysis"]
