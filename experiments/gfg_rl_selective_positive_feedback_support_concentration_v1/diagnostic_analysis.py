"""Post-hoc diagnostics for the single failed preregistered temporal gate.

This module never changes the frozen decision gates or scientific status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def skill_mean(evaluation: dict[str, Any], key: str, skills: list[int]) -> float:
    return mean(float(evaluation["per_skill"][skill][key]) for skill in skills)


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = (
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def run(artifact_root: Path) -> dict[str, Any]:
    formal = read_json(artifact_root / "FORMAL_RESULT.json")
    rows = []
    pooled_support: list[float] = []
    pooled_accuracy: list[float] = []
    within_correlations = []
    support_mass_rows = []
    for compact in formal["per_seed"]:
        seed = int(compact["seed"])
        result = read_json(artifact_root / f"seed-{seed}" / "SEED_RESULT.json")
        baseline_mass = result["baseline"]["support"]["task_support_mass"]
        selective_mass = result["conditions"]["selective"]["final_support"]["task_support_mass"]
        balanced_mass = result["conditions"]["balanced"]["final_support"]["task_support_mass"]
        support_mass_rows.append({
            "baseline_reinforced": float(baseline_mass[0]),
            "selective_reinforced": float(selective_mass[0]),
            "balanced_reinforced": float(balanced_mass[0]),
            "baseline_unreinforced": mean(float(value) for value in baseline_mass[1:]),
            "selective_unreinforced": mean(float(value) for value in selective_mass[1:]),
            "balanced_unreinforced": mean(float(value) for value in balanced_mass[1:]),
        })
        baseline_share = float(result["baseline"]["support"]["task_support_shares"][0])
        measurements = result["conditions"]["selective"]["measurements"]
        curve = []
        for measurement in measurements:
            support_delta = float(measurement["support"]["task_support_shares"][0]) - baseline_share
            unreinforced_accuracy = skill_mean(measurement["evaluation"], "chain_accuracy", [1, 2, 3])
            unreinforced_margin = skill_mean(measurement["evaluation"], "mean_margin", [1, 2, 3])
            curve.append({
                "update": int(measurement["update"]),
                "reinforced_support_share_delta": support_delta,
                "unreinforced_mean_chain_accuracy": unreinforced_accuracy,
                "unreinforced_mean_margin": unreinforced_margin,
            })
            if int(measurement["update"]) > 0:
                pooled_support.append(support_delta)
                pooled_accuracy.append(unreinforced_accuracy)
        nonzero = [row for row in curve if row["update"] > 0]
        first_positive = next(
            (row["update"] for row in nonzero if row["reinforced_support_share_delta"] > 1e-9),
            None,
        )
        first_drop = next(
            (row["update"] for row in nonzero if row["unreinforced_mean_chain_accuracy"] < 1.0 - 1e-12),
            None,
        )
        at_drop = next(row for row in curve if row["update"] == first_drop)
        directionals = [row for row in result["rollbacks"]["rows"] if row["directional_proper_subset"]]
        best = max(directionals, key=lambda row: row["unreinforced_mean_accuracy_change"])
        support_values = [row["reinforced_support_share_delta"] for row in nonzero]
        accuracy_values = [row["unreinforced_mean_chain_accuracy"] for row in nonzero]
        within = pearson(support_values, accuracy_values)
        within_correlations.append(within)
        rows.append({
            "seed": seed,
            "curve": curve,
            "first_any_positive_support_delta_update": first_positive,
            "first_accuracy_drop_update": first_drop,
            "any_positive_support_before_or_at_drop": first_positive is not None and first_drop is not None and first_positive <= first_drop,
            "support_share_delta_at_first_drop": at_drop["reinforced_support_share_delta"],
            "within_seed_support_share_vs_unreinforced_accuracy_pearson": within,
            "directional_proper_subset_count": len(directionals),
            "best_directional_rollback": {
                "mask": best["mask"],
                "components": best["components"],
                "unreinforced_mean_accuracy_change": best["unreinforced_mean_accuracy_change"],
                "reinforced_mean_margin_change": best["reinforced_mean_margin_change"],
            },
            "selective_hhi_increase_vs_baseline": (
                float(result["conditions"]["selective"]["final_support"]["cross_task_hhi"])
                - float(result["baseline"]["support"]["cross_task_hhi"])
            ),
            "selective_hhi_excess_vs_balanced": (
                float(result["conditions"]["selective"]["final_support"]["cross_task_hhi"])
                - float(result["conditions"]["balanced"]["final_support"]["cross_task_hhi"])
            ),
        })
    report = {
        "schema": "rl-e05-posthoc-temporal-diagnostic-v1",
        "classification": "DIAGNOSTIC_ONLY_NOT_A_REPLACEMENT_DECISION_GATE",
        "frozen_scientific_status": formal["scientific_status"],
        "reason_for_diagnostic": "the preregistered 0.03 support-share event threshold passed in only 2/12 seeds before or at the first accuracy drop",
        "seed_count": len(rows),
        "counts": {
            "any_positive_support_before_or_at_drop": sum(row["any_positive_support_before_or_at_drop"] for row in rows),
            "strictly_before_drop": sum(row["first_any_positive_support_delta_update"] < row["first_accuracy_drop_update"] for row in rows),
            "simultaneous_at_first_observed_checkpoint": sum(row["first_any_positive_support_delta_update"] == row["first_accuracy_drop_update"] for row in rows),
        },
        "means": {
            "support_share_delta_at_first_drop": mean(row["support_share_delta_at_first_drop"] for row in rows),
            "within_seed_support_share_vs_unreinforced_accuracy_pearson": mean(within_correlations),
            "pooled_support_share_vs_unreinforced_accuracy_pearson": pearson(pooled_support, pooled_accuracy),
            "directional_proper_subset_count_out_of_14": mean(row["directional_proper_subset_count"] for row in rows),
            "best_rollback_unreinforced_accuracy_gain": mean(row["best_directional_rollback"]["unreinforced_mean_accuracy_change"] for row in rows),
            "best_rollback_reinforced_margin_cost": mean(row["best_directional_rollback"]["reinforced_mean_margin_change"] for row in rows),
            "selective_hhi_increase_vs_baseline": mean(row["selective_hhi_increase_vs_baseline"] for row in rows),
            "selective_hhi_excess_vs_balanced": mean(row["selective_hhi_excess_vs_balanced"] for row in rows),
            "reinforced_positive_support_mass_increase_vs_baseline": mean(
                row["selective_reinforced"] - row["baseline_reinforced"] for row in support_mass_rows
            ),
            "unreinforced_positive_support_mass_change_vs_baseline": mean(
                row["selective_unreinforced"] - row["baseline_unreinforced"] for row in support_mass_rows
            ),
            "unreinforced_positive_support_mass_deficit_vs_balanced": mean(
                row["balanced_unreinforced"] - row["selective_unreinforced"] for row in support_mass_rows
            ),
        },
        "interpretation": (
            "At the available checkpoint resolution, a small positive support-share change was already present before "
            "or at every first behavioural drop, but the preregistered 0.03 event threshold was usually reached later. "
            "This diagnoses threshold/timescale mismatch; it does not convert the frozen NOT_SUPPORTED verdict to PASS."
        ),
        "per_seed": rows,
    }
    target = artifact_root / "DIAGNOSTIC_ONLY_TEMPORAL_ANALYSIS.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
