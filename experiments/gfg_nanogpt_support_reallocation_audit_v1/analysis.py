from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import FACTOR_RECORDS


COMPONENTS = ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")
ALPHAS = np.asarray((-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sections() -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    with gzip.open(FACTOR_RECORDS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                values.setdefault(str(row["section_id"]), row)
    require(len(values) == 72, "SECTION_COUNT")
    return [values[key] for key in sorted(values)]


def _float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _rho(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3 or len(np.unique(x[valid])) < 2 or len(np.unique(y[valid])) < 2:
        return None
    return float(spearmanr(x[valid], y[valid]).statistic)


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "maximum": float(np.max(x)),
    }


def run_analysis() -> dict[str, Any]:
    sections = _sections()
    group_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    source_ledger: list[dict[str, Any]] = []

    for record in sections:
        section_path = Path(record["source_refs"]["section_npz"])
        require(section_path.is_file(), f"SECTION_MISSING:{section_path}")
        require(sha256_file(section_path) == record["source_refs"]["section_npz_sha256"], "SECTION_HASH")
        with np.load(section_path, allow_pickle=False) as data:
            require(np.array_equal(np.asarray(data["alphas"], dtype=np.float64), ALPHAS), "ALPHA_GRID")
            groups = np.asarray(data["groups"], dtype=np.int64)
            predictions = np.asarray(data["all_predictions"], dtype=np.int64)
            allocation0 = np.asarray(data["support_allocation"][1].T, dtype=np.float64)
            allocation1 = np.asarray(data["support_allocation"][-1].T, dtype=np.float64)
            necessity0 = np.asarray(data["necessity"][1].T, dtype=np.float64)
            necessity1 = np.asarray(data["necessity"][-1].T, dtype=np.float64)
            backup0 = np.asarray(data["pair_backup"][1].T, dtype=np.float64)
            backup1 = np.asarray(data["pair_backup"][-1].T, dtype=np.float64)
            concentration0 = np.asarray(data["support_concentration"][1], dtype=np.float64)
            concentration1 = np.asarray(data["support_concentration"][-1], dtype=np.float64)
            effective0 = np.asarray(data["effective_support"][1], dtype=np.float64)
            effective1 = np.asarray(data["effective_support"][-1], dtype=np.float64)
            single0 = np.asarray(data["single_failure_slack"][1], dtype=np.float64)
            single1 = np.asarray(data["single_failure_slack"][-1], dtype=np.float64)
            double0 = np.asarray(data["double_failure_slack"][1], dtype=np.float64)
            double1 = np.asarray(data["double_failure_slack"][-1], dtype=np.float64)

        update = record["features"]["F3"]["numeric"]
        component_update = [float(update[f"{component}_update_l2"]) for component in COMPONENTS]
        update_l2 = float(update["update_l2_norm"])
        source_ledger.append(
            {
                "section_id": record["section_id"],
                "entry_id": record["entry_id"],
                "optimizer_step": int(record["optimizer_step"]),
                "section_npz": str(section_path),
                "section_npz_sha256": sha256_file(section_path),
                "transition": record["source_refs"]["transition"],
                "transition_sha256": record["source_refs"]["transition_sha256"],
            }
        )
        section_group_rows: list[dict[str, Any]] = []
        for group in range(23):
            allocation_valid = bool(
                np.all(np.isfinite(allocation0[group]))
                and np.all(np.isfinite(allocation1[group]))
                and abs(float(np.sum(allocation0[group])) - 1.0) <= 1e-6
                and abs(float(np.sum(allocation1[group])) - 1.0) <= 1e-6
            )
            selector = groups == group
            capability0 = float(np.mean(predictions[1, 0, selector] == group))
            capability1 = float(np.mean(predictions[-1, 0, selector] == group))
            delta_allocation = allocation1[group] - allocation0[group]
            delta_necessity = necessity1[group] - necessity0[group]
            row = {
                "section_id": record["section_id"],
                "entry_id": record["entry_id"],
                "optimizer_step": int(record["optimizer_step"]),
                "target_group": group,
                "update_l2_norm": update_l2,
                "component_update_l2": dict(zip(COMPONENTS, component_update)),
                "allocation_valid": allocation_valid,
                "allocation_before": dict(zip(COMPONENTS, allocation0[group].tolist())),
                "allocation_after": dict(zip(COMPONENTS, allocation1[group].tolist())),
                "allocation_delta": dict(zip(COMPONENTS, delta_allocation.tolist())),
                "necessity_before": dict(zip(COMPONENTS, necessity0[group].tolist())),
                "necessity_after": dict(zip(COMPONENTS, necessity1[group].tolist())),
                "necessity_delta": dict(zip(COMPONENTS, delta_necessity.tolist())),
                "pair_backup_before": backup0[group].tolist(),
                "pair_backup_after": backup1[group].tolist(),
                "pair_backup_delta": (backup1[group] - backup0[group]).tolist(),
                "support_concentration_before": _float(concentration0[group]),
                "support_concentration_after": _float(concentration1[group]),
                "support_concentration_delta": _float(concentration1[group] - concentration0[group]),
                "effective_support_before": _float(effective0[group]),
                "effective_support_after": _float(effective1[group]),
                "effective_support_delta": _float(effective1[group] - effective0[group]),
                "single_failure_slack_before": _float(single0[group]),
                "single_failure_slack_after": _float(single1[group]),
                "single_failure_slack_delta": _float(single1[group] - single0[group]),
                "double_failure_slack_before": _float(double0[group]),
                "double_failure_slack_after": _float(double1[group]),
                "double_failure_slack_delta": _float(double1[group] - double0[group]),
                "capability_before": capability0,
                "capability_after": capability1,
                "capability_delta": capability1 - capability0,
                "reallocation_magnitude": float(0.5 * np.sum(np.abs(delta_allocation))) if allocation_valid else None,
                "primary_support_before": COMPONENTS[int(np.argmax(allocation0[group]))] if allocation_valid else None,
                "primary_support_after": COMPONENTS[int(np.argmax(allocation1[group]))] if allocation_valid else None,
            }
            row["primary_support_switched"] = bool(
                allocation_valid and row["primary_support_before"] != row["primary_support_after"]
            )
            row["row_content_sha256"] = hashlib.sha256(canonical(row).encode("utf-8")).hexdigest()
            group_rows.append(row)
            section_group_rows.append(row)
        valid = [row for row in section_group_rows if row["allocation_valid"]]
        section_rows.append(
            {
                "section_id": record["section_id"],
                "entry_id": record["entry_id"],
                "optimizer_step": int(record["optimizer_step"]),
                "update_l2_norm": update_l2,
                "component_update_l2": dict(zip(COMPONENTS, component_update)),
                "valid_target_groups": len(valid),
                "mean_reallocation_magnitude": float(np.mean([row["reallocation_magnitude"] for row in valid])),
                "primary_support_switch_rate": float(np.mean([row["primary_support_switched"] for row in valid])),
                "mean_allocation_delta": {
                    component: float(np.mean([row["allocation_delta"][component] for row in valid]))
                    for component in COMPONENTS
                },
                "mean_necessity_delta": {
                    component: float(np.mean([row["necessity_delta"][component] for row in section_group_rows]))
                    for component in COMPONENTS
                },
            }
        )

    valid_rows = [row for row in group_rows if row["allocation_valid"]]
    changed = [row for row in valid_rows if row["capability_delta"] != 0.0]
    stable = [row for row in valid_rows if row["capability_delta"] == 0.0]
    per_run: dict[str, Any] = {}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_run[row["entry_id"]].append(row)
    for entry_id, rows in sorted(by_run.items()):
        per_run[entry_id] = {
            "count": len(rows),
            "update_to_reallocation_rho": _rho((row["update_l2_norm"] for row in rows), (row["reallocation_magnitude"] for row in rows)),
            "reallocation_to_absolute_capability_change_rho": _rho((row["reallocation_magnitude"] for row in rows), (abs(row["capability_delta"]) for row in rows)),
            "effective_support_to_capability_change_rho": _rho((row["effective_support_delta"] for row in rows), (row["capability_delta"] for row in rows)),
        }

    component_relations: dict[str, Any] = {}
    for component in COMPONENTS:
        component_relations[component] = {
            "update_to_mean_allocation_delta_rho": _rho(
                (row["component_update_l2"][component] for row in section_rows),
                (row["mean_allocation_delta"][component] for row in section_rows),
            ),
            "update_to_mean_necessity_delta_rho": _rho(
                (row["component_update_l2"][component] for row in section_rows),
                (row["mean_necessity_delta"][component] for row in section_rows),
            ),
        }

    summary = {
        "schema": "nanogpt-update-driven-support-reallocation-audit-v1",
        "status": "PASS",
        "claim_scope": "EXPLORATORY_POST_HOC_CAUSAL_DESCRIPTION_NOT_ADVANCE_PREDICTION",
        "section_count": len(section_rows),
        "entry_count": len(by_run),
        "target_group_transition_count": len(group_rows),
        "valid_allocation_transition_count": len(valid_rows),
        "invalid_allocation_transition_count": len(group_rows) - len(valid_rows),
        "capability_changed_count": len(changed),
        "capability_declined_count": sum(row["capability_delta"] < 0 for row in valid_rows),
        "capability_improved_count": sum(row["capability_delta"] > 0 for row in valid_rows),
        "capability_unchanged_count": len(stable),
        "reallocation_magnitude": _quantiles(row["reallocation_magnitude"] for row in valid_rows),
        "primary_support_switch_count": sum(row["primary_support_switched"] for row in valid_rows),
        "primary_support_switch_rate": float(np.mean([row["primary_support_switched"] for row in valid_rows])),
        "mean_reallocation_when_capability_changed": float(np.mean([row["reallocation_magnitude"] for row in changed])),
        "mean_reallocation_when_capability_unchanged": float(np.mean([row["reallocation_magnitude"] for row in stable])),
        "relations": {
            "update_magnitude_to_section_mean_reallocation_rho": _rho(
                (row["update_l2_norm"] for row in section_rows),
                (row["mean_reallocation_magnitude"] for row in section_rows),
            ),
            "update_magnitude_to_primary_support_switch_rate_rho": _rho(
                (row["update_l2_norm"] for row in section_rows),
                (row["primary_support_switch_rate"] for row in section_rows),
            ),
            "reallocation_to_capability_change_rho": _rho(
                (row["reallocation_magnitude"] for row in valid_rows),
                (row["capability_delta"] for row in valid_rows),
            ),
            "reallocation_to_absolute_capability_change_rho": _rho(
                (row["reallocation_magnitude"] for row in valid_rows),
                (abs(row["capability_delta"]) for row in valid_rows),
            ),
            "effective_support_change_to_capability_change_rho": _rho(
                (row["effective_support_delta"] for row in valid_rows),
                (row["capability_delta"] for row in valid_rows),
            ),
            "concentration_change_to_capability_change_rho": _rho(
                (row["support_concentration_delta"] for row in valid_rows),
                (row["capability_delta"] for row in valid_rows),
            ),
        },
        "component_relations": component_relations,
        "per_run": per_run,
        "future_information_used_as_predictor": False,
        "new_execution": False,
    }
    return {"summary": summary, "group_rows": group_rows, "section_rows": section_rows, "source_ledger": source_ledger}


__all__ = ["FACTOR_RECORDS", "run_analysis", "sha256_file"]
