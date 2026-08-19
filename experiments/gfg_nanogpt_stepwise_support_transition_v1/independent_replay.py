from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    payload_sha256,
    read_json,
    require,
    write_json,
)


def _write_frozen(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        require(read_json(path) == value, f"SST_INDEPENDENT_REPLAY_FROZEN_DRIFT:{path}")
        return
    write_json(path, value)
    require(read_json(path) == value, f"SST_INDEPENDENT_REPLAY_REREAD_MISMATCH:{path}")


def freeze_independent_replay_contract(
    *,
    selection_path: Path,
    pairing_path: Path,
    output_selection_path: Path,
    output_pairing_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = read_json(selection_path)
    pairing = read_json(pairing_path)
    selection_material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    pairing_material = {key: value for key, value in pairing.items() if key != "pairing_sha256"}
    require(payload_sha256(selection_material) == selection["selection_sha256"], "SST_INDEPENDENT_PARENT_SELECTION_INVALID")
    require(payload_sha256(pairing_material) == pairing["pairing_sha256"], "SST_INDEPENDENT_PARENT_PAIRING_INVALID")

    windows = {str(row["window_id"]): row for row in selection["windows"]}
    pairings = {str(row["pair_id"]): row for row in pairing["pairings"]}
    replay_rows = pairing["independent_replay_pairs"]
    require(bool(replay_rows), "SST_INDEPENDENT_REPLAY_PAIR_SET_EMPTY")
    replay_pair_ids = [str(row["pair_id"]) for row in replay_rows]
    require(len(set(replay_pair_ids)) == len(replay_pair_ids), "SST_INDEPENDENT_REPLAY_PAIR_DUPLICATE")
    require(set(replay_pair_ids) <= set(pairings), "SST_INDEPENDENT_REPLAY_PAIR_UNKNOWN")

    selected_pairings = [pairings[pair_id] for pair_id in replay_pair_ids]
    selected_window_ids = sorted(
        {
            str(pair[window_key])
            for pair in selected_pairings
            for window_key in ("decline_window_id", "stable_window_id")
        }
    )
    require(set(selected_window_ids) <= set(windows), "SST_INDEPENDENT_REPLAY_WINDOW_UNKNOWN")
    selected_windows = [windows[window_id] for window_id in selected_window_ids]
    require(
        {str(row["entry_id"]) for row in replay_rows}
        <= {str(row["entry_id"]) for row in selected_windows},
        "SST_INDEPENDENT_REPLAY_ENTRY_COVERAGE_INCOMPLETE",
    )

    selection_body = {
        "schema": "nanogpt-stepwise-independent-replay-window-selection-v1",
        "status": "FROZEN_FROM_PREDECLARED_REPLAY_PAIRS_BEFORE_FORMAL_V3_RESULTS_COMPLETE",
        "protocol_sha256": selection["protocol_sha256"],
        "finite_difference_protocol_sha256": selection[
            "finite_difference_protocol_sha256"
        ],
        "parent_selection_sha256": selection["selection_sha256"],
        "parent_pairing_sha256": pairing["pairing_sha256"],
        "result_dependent_selection_used": False,
        "selection_rule": "exact union of decline and matched-stable windows named by independent_replay_pairs",
        "pair_ids": replay_pair_ids,
        "window_count": len(selected_windows),
        "windows": selected_windows,
    }
    replay_selection = {**selection_body, "selection_sha256": payload_sha256(selection_body)}

    pairing_body = {
        "schema": "nanogpt-stepwise-independent-replay-pairing-v1",
        "status": "FROZEN_FROM_PREDECLARED_REPLAY_PAIRS_BEFORE_FORMAL_V3_RESULTS_COMPLETE",
        "protocol_sha256": selection["protocol_sha256"],
        "parent_selection_sha256": selection["selection_sha256"],
        "parent_pairing_sha256": pairing["pairing_sha256"],
        "replay_selection_sha256": replay_selection["selection_sha256"],
        "result_dependent_selection_used": False,
        "independent_replay_pairs": replay_rows,
        "pairings": selected_pairings,
    }
    replay_pairing = {**pairing_body, "pairing_sha256": payload_sha256(pairing_body)}
    _write_frozen(output_selection_path, replay_selection)
    _write_frozen(output_pairing_path, replay_pairing)
    return replay_selection, replay_pairing


def freeze_independent_branch_replay_audit(
    *,
    divergence_audit_path: Path,
    replay_pairing_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    audit = read_json(divergence_audit_path)
    replay_pairing = read_json(replay_pairing_path)
    _verify_fields = (
        (audit, "audit_sha256", "SST_INDEPENDENT_BRANCH_PARENT_AUDIT_INVALID"),
        (replay_pairing, "pairing_sha256", "SST_INDEPENDENT_BRANCH_PAIRING_INVALID"),
    )
    for value, hash_field, error in _verify_fields:
        material = {key: child for key, child in value.items() if key != hash_field}
        require(payload_sha256(material) == value[hash_field], error)
    replay_pair_ids = {str(row["pair_id"]) for row in replay_pairing["independent_replay_pairs"]}
    selected = [row for row in audit["results"] if str(row["pair_id"]) in replay_pair_ids]
    require({str(row["pair_id"]) for row in selected} == replay_pair_ids, "SST_INDEPENDENT_BRANCH_AUDIT_PAIR_COVERAGE_INCOMPLETE")
    body = {
        "schema": "nanogpt-stepwise-independent-branch-replay-audit-v1",
        "status": "PASS",
        "source_divergence_audit_sha256": audit["audit_sha256"],
        "replay_pairing_sha256": replay_pairing["pairing_sha256"],
        "selection_rule": "all frozen causal seed offsets for every predeclared independent replay pair",
        "result_dependent_branch_result_selection_used": False,
        "results": selected,
    }
    result = {**body, "audit_sha256": payload_sha256(body)}
    _write_frozen(output_path, result)
    return result
