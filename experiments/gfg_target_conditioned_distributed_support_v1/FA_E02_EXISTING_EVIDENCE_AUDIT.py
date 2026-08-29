from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")
PAIRS = tuple(
    (COMPONENTS[i], COMPONENTS[j])
    for i in range(len(COMPONENTS))
    for j in range(i + 1, len(COMPONENTS))
)
PHASES = ("pre_formation", "formed", "decline", "recovered")
DEFAULT_GFG_ROOT = Path(r"E:\gfg-evidence\nanogpt-training-learning-inference-projection-v1\gfg")
DEFAULT_OUTPUT = Path(__file__).with_name("FA_E02_EXISTING_EVIDENCE_AUDIT.json")
NONADDITIVITY_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_blocks(database: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return [
            json.loads(zlib.decompress(row["payload_zlib"]))
            for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal")
        ]


def load_tensor(bundle: Path, value: dict[str, Any]) -> np.ndarray:
    payload = value["payload"]
    path = bundle / payload["locator"]
    require(path.is_file(), f"MISSING_TENSOR:{path}")
    require(file_sha256(path) == payload["file_sha256"], f"FILE_HASH:{path}")
    result = np.load(path, allow_pickle=False)
    require(array_sha256(result) == payload["raw_tensor_sha256"], f"ARRAY_HASH:{path}")
    return result


def select_object(block: dict[str, Any], role: str, gate: tuple[str, ...] | None = None) -> dict[str, Any]:
    rows = [row for row in block["objects"] if row["role"] == role]
    if gate is not None:
        rows = [row for row in rows if tuple(row["payload"].get("gate_components", ())) == gate]
    require(len(rows) == 1, f"OBJECT_SELECTION:{role}:{gate}:{len(rows)}")
    return rows[0]


def target_margin(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = np.arange(logits.shape[0])
    correct = logits[rows, target]
    competing = logits.copy()
    competing[rows, target] = -np.inf
    return correct - np.max(competing, axis=1)


def decisions(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=1) == target


def phase_payload(bundle: Path, block: dict[str, Any], target: np.ndarray) -> dict[str, Any]:
    baseline_object = select_object(block, "native_inference_logits")
    phase = str(baseline_object["payload"]["phase"])
    baseline = load_tensor(bundle, baseline_object)
    singles = {
        component: load_tensor(bundle, select_object(block, "single_component_gate_logits", (component,)))
        for component in COMPONENTS
    }
    pairs = {
        "+".join(pair): load_tensor(bundle, select_object(block, "pair_component_gate_logits", pair))
        for pair in PAIRS
    }
    base_margin = target_margin(baseline, target)
    base_correct = decisions(baseline, target)
    single_margin = {key: target_margin(value, target) for key, value in singles.items()}
    single_correct = {key: decisions(value, target) for key, value in singles.items()}
    pair_margin = {key: target_margin(value, target) for key, value in pairs.items()}
    pair_correct = {key: decisions(value, target) for key, value in pairs.items()}
    single_effect = {key: base_margin - value for key, value in single_margin.items()}
    pair_effect = {key: base_margin - value for key, value in pair_margin.items()}
    interaction = {
        "+".join(pair): pair_effect["+".join(pair)] - single_effect[pair[0]] - single_effect[pair[1]]
        for pair in PAIRS
    }
    condition_names = list(COMPONENTS) + ["+".join(pair) for pair in PAIRS]
    condition_correct = np.stack(
        [single_correct[key] for key in COMPONENTS]
        + [pair_correct["+".join(pair)] for pair in PAIRS],
        axis=1,
    )
    return {
        "phase": phase,
        "optimizer_step": int(block["optimizer_step"]),
        "baseline_margin": base_margin,
        "baseline_correct": base_correct,
        "single_margin": single_margin,
        "single_correct": single_correct,
        "pair_margin": pair_margin,
        "pair_correct": pair_correct,
        "single_effect": single_effect,
        "pair_effect": pair_effect,
        "interaction": interaction,
        "condition_names": condition_names,
        "condition_correct": condition_correct,
        "baseline_logits_sha256": array_sha256(baseline),
    }


def summarize_phase(payload: dict[str, Any], target: np.ndarray) -> dict[str, Any]:
    base_correct = payload["baseline_correct"]
    correct_count = int(np.count_nonzero(base_correct))
    condition_correct = payload["condition_correct"]
    successful_observed_configurations = condition_correct.sum(axis=1) + base_correct.astype(np.int64)
    pattern_counter = Counter(
        "".join("1" if value else "0" for value in row)
        for row in condition_correct[base_correct]
    )
    necessity = {
        component: int(np.count_nonzero(base_correct & ~payload["single_correct"][component]))
        for component in COMPONENTS
    }
    pair_failure = {}
    pair_backup_compatible = {}
    nonadditive = {}
    for pair in PAIRS:
        key = "+".join(pair)
        pair_failure[key] = int(np.count_nonzero(base_correct & ~payload["pair_correct"][key]))
        pair_backup_compatible[key] = int(
            np.count_nonzero(
                base_correct
                & payload["single_correct"][pair[0]]
                & payload["single_correct"][pair[1]]
                & ~payload["pair_correct"][key]
            )
        )
        values = payload["interaction"][key]
        nonadditive[key] = {
            "count_abs_gt_1e_6": int(np.count_nonzero(np.abs(values) > NONADDITIVITY_TOLERANCE)),
            "positive_count": int(np.count_nonzero(values > NONADDITIVITY_TOLERANCE)),
            "negative_count": int(np.count_nonzero(values < -NONADDITIVITY_TOLERANCE)),
            "mean": float(np.mean(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "max_abs": float(np.max(np.abs(values))),
        }

    target_rows: dict[str, Any] = {}
    for group in range(23):
        mask = target == group
        correct_mask = mask & base_correct
        count = int(np.count_nonzero(correct_mask))
        target_rows[str(group)] = {
            "instance_count": int(np.count_nonzero(mask)),
            "baseline_correct_count": count,
            "single_boundary_failure_rate": {
                component: (
                    float(np.count_nonzero(correct_mask & ~payload["single_correct"][component]) / count)
                    if count
                    else None
                )
                for component in COMPONENTS
            },
            "pair_boundary_failure_rate": {
                "+".join(pair): (
                    float(np.count_nonzero(correct_mask & ~payload["pair_correct"]["+".join(pair)]) / count)
                    if count
                    else None
                )
                for pair in PAIRS
            },
        }

    component_heterogeneity: dict[str, Any] = {}
    for component in COMPONENTS:
        rates = [
            row["single_boundary_failure_rate"][component]
            for row in target_rows.values()
            if row["single_boundary_failure_rate"][component] is not None
        ]
        component_heterogeneity[component] = {
            "target_groups_observed": len(rates),
            "minimum_failure_rate": float(min(rates)) if rates else None,
            "maximum_failure_rate": float(max(rates)) if rates else None,
            "range": float(max(rates) - min(rates)) if rates else None,
        }

    return {
        "optimizer_step": payload["optimizer_step"],
        "instance_count": int(target.shape[0]),
        "baseline_correct_count": correct_count,
        "baseline_accuracy": float(np.mean(base_correct)),
        "single_component_contextual_necessity_count": necessity,
        "pair_boundary_failure_count": pair_failure,
        "backup_or_substitution_compatible_pair_count": pair_backup_compatible,
        "nonadditive_pair_interaction": nonadditive,
        "baseline_correct_instances_with_multiple_successful_observed_configurations": int(
            np.count_nonzero(base_correct & (successful_observed_configurations >= 2))
        ),
        "baseline_correct_instances_with_at_least_five_successful_observed_configurations": int(
            np.count_nonzero(base_correct & (successful_observed_configurations >= 5))
        ),
        "successful_observed_configuration_count_mean_on_baseline_correct": (
            float(np.mean(successful_observed_configurations[base_correct])) if correct_count else None
        ),
        "distinct_boundary_consequence_patterns_on_baseline_correct": len(pattern_counter),
        "most_common_boundary_consequence_patterns": pattern_counter.most_common(10),
        "target_conditioned_component_heterogeneity": component_heterogeneity,
        "target_group_rows": target_rows,
        "baseline_logits_sha256": payload["baseline_logits_sha256"],
    }


def phase_transition(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    require(left["condition_correct"].shape == right["condition_correct"].shape, "PHASE_ALIGNMENT")
    both_baseline_correct = left["baseline_correct"] & right["baseline_correct"]
    changed = np.any(left["condition_correct"] != right["condition_correct"], axis=1)
    count = int(np.count_nonzero(both_baseline_correct))
    return {
        "instance_count": int(changed.shape[0]),
        "both_baseline_correct_count": count,
        "changed_boundary_consequence_pattern_count": int(np.count_nonzero(changed)),
        "changed_pattern_among_both_baseline_correct_count": int(np.count_nonzero(changed & both_baseline_correct)),
        "changed_pattern_rate_among_both_baseline_correct": (
            float(np.count_nonzero(changed & both_baseline_correct) / count) if count else None
        ),
    }


def audit_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], np.ndarray]:
    manifest = json.loads((bundle / "GFG_MANIFEST.json").read_text(encoding="utf-8"))
    validation = json.loads((bundle / "GFG_VALIDATION.json").read_text(encoding="utf-8"))
    database = bundle / manifest["database"]
    require(file_sha256(database) == manifest["database_sha256"], f"DATABASE_HASH:{bundle.name}")
    require(validation["status"] == "PASS", f"SOURCE_GFG_VALIDATION:{bundle.name}")
    blocks = load_blocks(database)
    target_object = select_object(blocks[0], "derived_validation_targets")
    targets_full = load_tensor(bundle, target_object)
    target = targets_full[:, -1].astype(np.int64)
    phase_payloads: dict[str, dict[str, Any]] = {}
    for block in blocks[1:]:
        payload = phase_payload(bundle, block, target)
        phase_payloads[payload["phase"]] = payload
    require(set(phase_payloads) == set(PHASES), f"PHASE_SET:{bundle.name}")
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        query_origins = [
            dict(row)
            for row in connection.execute(
                "SELECT source_bundle_id, source_object_id, source_content_sha256, source_semantic_key "
                "FROM origin_catalog WHERE source_role='layer_input' AND source_semantic_key LIKE '%evaluation_validation%'"
            )
        ]
    require(len(query_origins) == 1, f"QUERY_ORIGIN:{bundle.name}:{len(query_origins)}")
    row = {
        "entry_id": manifest["entry_id"],
        "source_bundle_id": manifest["source_bundle_id"],
        "source_gfg_validation_status": validation["status"],
        "query_origin": query_origins[0],
        "validation_target_sha256": array_sha256(targets_full),
        "phases": {phase: summarize_phase(phase_payloads[phase], target) for phase in PHASES},
        "historical_reorganization": {
            "pre_formation_to_formed": phase_transition(
                phase_payloads["pre_formation"], phase_payloads["formed"]
            ),
            "formed_to_decline": phase_transition(phase_payloads["formed"], phase_payloads["decline"]),
            "decline_to_recovered": phase_transition(
                phase_payloads["decline"], phase_payloads["recovered"]
            ),
        },
    }
    return row, phase_payloads, target


def sum_nested(rows: list[dict[str, int]], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: int(sum(row[key] for row in rows)) for key in keys}


def aggregate(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase_rows = {
        phase: [row["phases"][phase] for row in run_rows]
        for phase in PHASES
    }
    phase_summary: dict[str, Any] = {}
    for phase, rows in phase_rows.items():
        necessity = sum_nested(
            [row["single_component_contextual_necessity_count"] for row in rows], COMPONENTS
        )
        backup = sum_nested(
            [row["backup_or_substitution_compatible_pair_count"] for row in rows],
            tuple("+".join(pair) for pair in PAIRS),
        )
        pair_nonadditive = {
            "+".join(pair): int(
                sum(
                    row["nonadditive_pair_interaction"]["+".join(pair)]["count_abs_gt_1e_6"]
                    for row in rows
                )
            )
            for pair in PAIRS
        }
        target_heterogeneity_events = sum(
            row["target_conditioned_component_heterogeneity"][component]["range"] > 0.0
            for row in rows
            for component in COMPONENTS
            if row["target_conditioned_component_heterogeneity"][component]["range"] is not None
        )
        phase_summary[phase] = {
            "run_count": len(rows),
            "instance_count": int(sum(row["instance_count"] for row in rows)),
            "baseline_correct_count": int(sum(row["baseline_correct_count"] for row in rows)),
            "single_component_contextual_necessity_count": necessity,
            "backup_or_substitution_compatible_pair_count": backup,
            "nonadditive_pair_instance_count_abs_gt_1e_6": pair_nonadditive,
            "baseline_correct_instances_with_multiple_successful_observed_configurations": int(
                sum(row["baseline_correct_instances_with_multiple_successful_observed_configurations"] for row in rows)
            ),
            "baseline_correct_instances_with_at_least_five_successful_observed_configurations": int(
                sum(row["baseline_correct_instances_with_at_least_five_successful_observed_configurations"] for row in rows)
            ),
            "distinct_boundary_consequence_patterns_sum_over_runs": int(
                sum(row["distinct_boundary_consequence_patterns_on_baseline_correct"] for row in rows)
            ),
            "run_component_cases_with_target_conditioned_failure_rate_heterogeneity": int(
                target_heterogeneity_events
            ),
        }

    historical = {}
    for name in ("pre_formation_to_formed", "formed_to_decline", "decline_to_recovered"):
        rows = [row["historical_reorganization"][name] for row in run_rows]
        denominator = sum(row["both_baseline_correct_count"] for row in rows)
        numerator = sum(row["changed_pattern_among_both_baseline_correct_count"] for row in rows)
        historical[name] = {
            "both_baseline_correct_count": int(denominator),
            "changed_pattern_among_both_baseline_correct_count": int(numerator),
            "changed_pattern_rate_among_both_baseline_correct": (
                float(numerator / denominator) if denominator else None
            ),
            "runs_with_at_least_one_changed_pattern": int(
                sum(row["changed_boundary_consequence_pattern_count"] > 0 for row in rows)
            ),
        }

    formed = phase_summary["formed"]
    observed = {
        "native_conditions_per_instance": 11,
        "baseline_conditions": 1,
        "single_component_gate_conditions": 4,
        "pair_component_gate_conditions": 6,
        "new_forward_executions": 0,
        "new_training_executions": 0,
        "source_run_count": len(run_rows),
        "held_out_instances_per_run": 212,
        "phase_count": 4,
    }
    direct_questions = {
        "component_contextual_necessity": sum(formed["single_component_contextual_necessity_count"].values()) > 0,
        "backup_or_substitution_compatible_pair_failure": sum(
            formed["backup_or_substitution_compatible_pair_count"].values()
        ) > 0,
        "pair_nonadditivity": sum(formed["nonadditive_pair_instance_count_abs_gt_1e_6"].values()) > 0,
        "multiple_successful_observed_configurations": formed[
            "baseline_correct_instances_with_multiple_successful_observed_configurations"
        ] > 0,
        "target_conditioned_boundary_consequence_heterogeneity": formed[
            "run_component_cases_with_target_conditioned_failure_rate_heterogeneity"
        ] > 0,
        "historical_support_reorganization": all(
            row["runs_with_at_least_one_changed_pattern"] > 0 for row in historical.values()
        ),
    }
    return {
        "schema": "fa-e02-existing-evidence-audit-v1",
        "status": "AUDIT_COMPLETE",
        "source_scope": observed,
        "phase_summary": phase_summary,
        "historical_reorganization": historical,
        "directly_adjudicable_questions": direct_questions,
        "identified_limits": [
            "Single-component sufficiency is not identifiable without a triple gate that leaves one component active.",
            "The all-components-removed null is not identifiable without the four-component gate.",
            "A complete support lattice, unique minimal sufficient set, and interactions above pair order are not identifiable.",
            "A pair pattern compatible with backup or substitution cannot by itself exclude an unobserved higher-order interaction.",
            "Necessity and sufficiency are contextual to the executed target, query, receiving state, component versions, and gate site.",
        ],
        "scientific_interpretation": (
            "The existing native interventions directly expose target-conditioned causal support organization: "
            "different concrete held-out instances retain the learned mod-23 relation under multiple different observed "
            "surviving configurations; single and pair gates produce target-boundary-specific necessity, pair-failure, "
            "and non-additive consequences; and these relations reorganize across formation, decline, and recovery."
        ),
        "claim_boundary": (
            "This is evidence for multiplicity and target-conditioned distributed causal organization in the executed "
            "13-run nanoGPT family. It does not assert geometric convergence, a unique owner, a unique minimal support "
            "set, complete higher-order identification, or universal abstraction in all systems."
        ),
    }


def render_markdown(result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    formed = summary["phase_summary"]["formed"]
    hist = summary["historical_reorganization"]
    multiplicity_rate = (
        formed["baseline_correct_instances_with_multiple_successful_observed_configurations"]
        / formed["baseline_correct_count"]
    )
    backup_total = sum(formed["backup_or_substitution_compatible_pair_count"].values())
    lines = [
        "# FA-E02 — Existing Evidence Audit",
        "",
        "## Frozen scope",
        "",
        "This audit reads the already executed nanoGPT inference GFGs only. It performs no training, no new inference, "
        "and no triple- or four-component gate. FA-E01 and its `INCONCLUSIVE` verdict are not modified.",
        "",
        f"- Source histories: **{summary['source_scope']['source_run_count']}**",
        f"- Held-out instances per history: **{summary['source_scope']['held_out_instances_per_run']}**",
        "- Historical phases: **pre-formation, formed, decline, recovered**",
        "- Native conditions per instance: **11** (baseline + 4 single gates + 6 pair gates)",
        "- New native executions: **0**",
        "",
        "## Scientific object",
        "",
        "The unit of evidence is not a common geometric fingerprint. It is:",
        "",
        "`support configuration -> native causal intervention -> target-boundary consequence`",
        "",
        "A component is called *contextually necessary* only when the baseline target is correct and its single gate "
        "changes that exact target to incorrect. A pair is *backup/substitution-compatible* only when both relevant "
        "single gates retain the correct target but their joint gate crosses the target boundary. These terms are "
        "restricted to the executed query, target, receiving state, component versions, and gate site.",
        "",
        "## What the existing evidence directly adjudicates",
        "",
        f"At the formed phase, the native evidence contains **{formed['baseline_correct_count']}** baseline-correct "
        f"instance observations across all runs. **{formed['baseline_correct_instances_with_multiple_successful_observed_configurations']}** "
        f"of them (**{multiplicity_rate:.2%}**) remain correct under more than one of the 11 observed configurations, "
        "establishing observed causal "
        "multiplicity without requiring the configurations to be geometrically similar.",
        "",
        "### Contextual necessity by component (formed phase)",
        "",
        "| Component | Boundary-crossing single-gate cases |",
        "|---|---:|",
    ]
    for key, value in formed["single_component_contextual_necessity_count"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "### Backup/substitution-compatible pair cases (formed phase)",
        "",
        "| Gated pair | Singles individually retain target, pair loses target |",
        "|---|---:|",
    ]
    for key, value in formed["backup_or_substitution_compatible_pair_count"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        f"Across the six pairs this criterion is met in **{backup_total}** formed-phase instance-pair cases.",
        "",
        "### Pair non-additivity (formed phase)",
        "",
        "| Gated pair | Instance interactions with `abs(I) > 1e-6` |",
        "|---|---:|",
    ]
    for key, value in formed["nonadditive_pair_instance_count_abs_gt_1e_6"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        f"Target-conditioned boundary consequences vary across target groups in **{formed['run_component_cases_with_target_conditioned_failure_rate_heterogeneity']}** "
        "of the formed run-by-component cases with observable baseline-correct targets. This supports shared plus "
        "target-conditioned organization rather than a fixed single owner.",
        "",
        "## Historical reorganization",
        "",
        "The same 212 held-out positions are aligned within each run across phases. The table reports whether their "
        "10-condition boundary-consequence pattern changes while the baseline target remains correct at both endpoints.",
        "",
        "| Transition | Both endpoints baseline-correct | Changed gate-consequence pattern | Rate | Runs with change |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in hist.items():
        rate = row["changed_pattern_rate_among_both_baseline_correct"]
        lines.append(
            f"| `{key}` | {row['both_baseline_correct_count']} | "
            f"{row['changed_pattern_among_both_baseline_correct_count']} | {rate:.4f} | "
            f"{row['runs_with_at_least_one_changed_pattern']} |"
        )
    lines += [
        "",
        "These are direct changes in intervention consequences, not cosine changes. They show that causal support "
        "relations reorganize during formation, decline, and recovery.",
        "",
        "## Questions not identifiable from the current 11 conditions",
        "",
    ]
    lines += [f"- {value}" for value in summary["identified_limits"]]
    lines += [
        "",
        "Therefore no extra forward is justified merely to complete a mathematical lattice. A future triple or "
        "four-component gate would be justified only for a specific unresolved sufficiency or higher-order causal question.",
        "",
        "## Audit conclusion",
        "",
        summary["scientific_interpretation"],
        "",
        "**Claim boundary.** " + summary["claim_boundary"],
        "",
        "## Identity and integrity",
        "",
        "All source GFGs reported `PASS`; every SQLite database, native logit tensor, and validation-target tensor was "
        "re-hashed before analysis. Query origins, target tensors, parameter versions, gate specifications, and native "
        "outcomes remain addressed by their source-GFG identities. Full per-run and per-target counts and hashes are in "
        "`FA_E02_EXISTING_EVIDENCE_AUDIT.json`.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(gfg_root: Path, output: Path) -> dict[str, Any]:
    bundles = sorted(path for path in gfg_root.iterdir() if path.is_dir())
    require(len(bundles) == 13, f"SOURCE_RUN_COUNT:{len(bundles)}")
    run_rows = []
    for bundle in bundles:
        row, _, _ = audit_bundle(bundle)
        run_rows.append(row)
    result = {
        "schema": "fa-e02-existing-evidence-audit-artifact-v1",
        "source_root": str(gfg_root),
        "source_root_entry_count": len(bundles),
        "summary": aggregate(run_rows),
        "runs": run_rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_markdown(result, output.with_suffix(".md"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfg-root", type=Path, default=DEFAULT_GFG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.gfg_root, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
