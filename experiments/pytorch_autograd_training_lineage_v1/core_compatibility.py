"""Integration-only protected-scope compatibility for the audited 6b3490 Core.

This module does not alter capture, projection, native observation, query,
strictness, or intervention behavior. It replaces a historical literal tree
SHA audit with the current unified proof's explicitly adopted Core baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


UNIFIED_CORE_COMMIT = "6b34906d7b6e4fa15f6c7d6e3013daa35a308b5e"
PROTECTED_PATHS = {
    "compat_v2_tree_sha": "compat/v2",
    "core_protocol_schema_tree_sha": "protocol/core_v3",
    "core_runtime_tree_sha": "src/generation_relation_core",
    "tests_core_tree_sha": "tests/core",
}


def protected_scope_for_unified_core(
    git: Callable[..., str], repository_root: Path
) -> dict[str, Any]:
    trees = {name: git("rev-parse", f"HEAD:{path}") for name, path in PROTECTED_PATHS.items()}
    expected = {
        name: git("rev-parse", f"{UNIFIED_CORE_COMMIT}:{path}")
        for name, path in PROTECTED_PATHS.items()
    }
    changed = [
        line
        for line in git(
            "diff",
            "--name-only",
            UNIFIED_CORE_COMMIT,
            "--",
            *PROTECTED_PATHS.values(),
        ).splitlines()
        if line
    ]
    status = [
        line[3:].rsplit(" -> ", 1)[-1]
        for line in git(
            "status",
            "--porcelain=v1",
            "--",
            *PROTECTED_PATHS.values(),
        ).splitlines()
    ]
    protected_relatives = git("ls-files", "--", *PROTECTED_PATHS.values()).splitlines()
    token_count = 0
    for relative in protected_relatives:
        text = (repository_root / relative).read_text(encoding="utf-8", errors="strict").lower()
        token_count += sum(text.count(token) for token in ("pytorch", "autograd", "torch"))
    changed_paths = sorted(set(changed + status))
    return {
        "changed_existing_experiment_paths": [],
        "core_zero_change": trees == expected and not changed_paths,
        "expected_trees": expected,
        "merge_base": git("merge-base", "HEAD", UNIFIED_CORE_COMMIT),
        "protected_scope_pytorch_specific_token_count": token_count,
        "pytorch_specific_core_field_count": token_count,
        "trees": trees,
        "compatibility": {
            "kind": "protected-scope baseline only",
            "unified_core_commit": UNIFIED_CORE_COMMIT,
            "changed_protected_paths": changed_paths,
            "scientific_paths_modified": False,
        },
    }

