from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import require
from experiments.gfg_nanogpt_support_transition_v1.storage import TensorStore


def load_named_tensor_ref(payload_root: Path, reference: dict[str, Any]) -> dict[str, torch.Tensor]:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_NAMED_TENSOR_LOCATOR_INVALID")
    packed = np.load(payload_root / locator, allow_pickle=False, mmap_mode="r")
    result: dict[str, torch.Tensor] = {}
    for row in reference["layout"]:
        offset = int(row["offset"])
        count = int(row["element_count"])
        child = np.asarray(packed[offset : offset + count]).astype(np.dtype(row["dtype"]), copy=True)
        child = child.reshape(tuple(int(value) for value in row["shape"]))
        result[str(row["name"])] = torch.from_numpy(child)
    require(tuple(sorted(result)) == tuple(reference["canonical_name_order"]), "SST_NAMED_TENSOR_LAYOUT_INVALID")
    return result


def restorable_state_from_manifest(payload_root: Path, manifest: dict[str, Any]):
    from experiments.gfg_nanogpt_support_transition_v1.runtime import StateSnapshot

    parameters = load_named_tensor_ref(payload_root, manifest["parameters"])
    exp_avg = load_named_tensor_ref(payload_root, manifest["optimizer_exp_avg"])
    exp_avg_sq = load_named_tensor_ref(payload_root, manifest["optimizer_exp_avg_sq"])
    steps = load_named_tensor_ref(payload_root, manifest["optimizer_steps"])
    require(set(parameters) == set(exp_avg) == set(exp_avg_sq) == set(steps), "SST_RESTORABLE_STATE_NAME_MISMATCH")
    optimizer = {
        name: {
            "step": steps[name],
            "exp_avg": exp_avg[name],
            "exp_avg_sq": exp_avg_sq[name],
        }
        for name in parameters
    }
    state = StateSnapshot(parameters=parameters, optimizer=optimizer)
    require(state.commitment() == manifest["commitment"], "SST_RESTORABLE_STATE_COMMITMENT_MISMATCH")
    return state


__all__ = ["TensorStore", "load_named_tensor_ref", "restorable_state_from_manifest"]
