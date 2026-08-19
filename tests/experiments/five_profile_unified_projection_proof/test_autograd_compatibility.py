from __future__ import annotations

import ast
import inspect
from pathlib import Path

from experiments.pytorch_autograd_training_lineage_v1 import core_compatibility
from experiments.pytorch_autograd_training_lineage_v1.science import _protected_scope
from experiments.five_profile_unified_projection_proof.src.mechanism_entry import CORE_COMMIT


REPO = Path(__file__).resolve().parents[3]


def test_compatibility_shim_is_audit_only_and_targets_adopted_core():
    source = inspect.getsource(core_compatibility)
    assert core_compatibility.UNIFIED_CORE_COMMIT == CORE_COMMIT
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"run_training_step", "project_snapshot", "run_complete_science", "build_strict_projection_counterexamples"}


def test_compatibility_shim_observes_zero_current_core_changes():
    scope = _protected_scope()
    assert scope["core_zero_change"]
    assert scope["compatibility"]["changed_protected_paths"] == []
    assert scope["pytorch_specific_core_field_count"] == 0
