from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from .common import canonical_sha256, file_sha256
from .mechanism_entry import CORE_COMMIT, SOURCE_COMMITS


PROTECTED_PREFIXES = (
    "src/generation_relation_core/",
    "protocol/core_v3/",
    "compat/v2/",
    "tests/core/",
)

PROFILE_DOCUMENTS = {
    "database_which_lineage": {
        "profile": "experiments/operational_projection_proof_v2/profiles/database_which_lineage_v1.json",
        "crosswalk": "experiments/operational_projection_proof_v2/profiles/database_which_lineage_v1.json",
        "crosswalk_location": "embedded in frozen profile",
    },
    "source_map": {
        "profile": "experiments/operational_projection_proof_v2/profiles/ecma426_ordinary_source_map_v1.json",
        "crosswalk": "experiments/operational_projection_proof_v2/profiles/ecma426_ordinary_source_map_v1.json",
        "crosswalk_location": "embedded in frozen profile",
    },
    "opentelemetry": {
        "profile": "experiments/operational_projection_proof_v2/profiles/opentelemetry_occurrence_execution_v1.json",
        "crosswalk": "experiments/operational_projection_proof_v2/profiles/opentelemetry_occurrence_execution_v1.json",
        "crosswalk_location": "embedded in frozen profile",
    },
    "w3c_prov_generation_profile": {
        "profile": "experiments/w3c_prov_projection_v1/profiles/w3c_prov_generation_profile_v1.json",
        "crosswalk": "experiments/w3c_prov_projection_v1/profiles/core_to_w3c_prov_crosswalk_v1.json",
        "crosswalk_location": "separate frozen crosswalk",
    },
    "pytorch_autograd_dependency_profile": {
        "profile": "experiments/pytorch_autograd_training_lineage_v1/profiles/pytorch_autograd_dependency_profile_v1.json",
        "crosswalk": "experiments/pytorch_autograd_training_lineage_v1/profiles/core_to_pytorch_autograd_crosswalk_v1.json",
        "crosswalk_location": "separate frozen crosswalk",
        "hardened_profile": "experiments/pytorch_autograd_training_lineage_v1/profiles/pytorch_gradient_value_dependency_profile_v1.json",
    },
}

ENTRYPOINTS = {
    "database_which_lineage": "experiments.operational_projection_proof_v2.src.database_proof.run_database_proof",
    "source_map": "experiments.operational_projection_proof_v2.src.source_map_proof.run_source_map_proof",
    "opentelemetry": "experiments.operational_projection_proof_v2.src.otel_proof.run_otel_proof",
    "w3c_prov_generation_profile": "experiments.w3c_prov_projection_v1.src.science_runs.run_full + strict_projection_counterexamples + oracle process audit",
    "pytorch_autograd_dependency_profile": "experiments.pytorch_autograd_training_lineage_v1.science.run_complete_science + hardening_science.run_complete_hardening_science",
}


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _node_version(repo: Path) -> str | None:
    from experiments.source_map_projection.src.node_bridge import find_node

    process = subprocess.run([str(find_node()), "--version"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout.strip() if process.returncode == 0 else None


def _core_changed_paths(repo: Path) -> list[str]:
    changed = _git(repo, "diff", "--name-only", CORE_COMMIT, "--", *PROTECTED_PREFIXES).splitlines()
    untracked_or_modified = []
    for line in _git(repo, "status", "--porcelain=v1", "--", *PROTECTED_PREFIXES).splitlines():
        untracked_or_modified.append(line[3:].rsplit(" -> ", 1)[-1])
    return sorted(set(filter(None, [*changed, *untracked_or_modified])))


def _tree_file_hash(repo: Path, prefixes: tuple[str, ...]) -> str:
    rows = []
    for prefix in prefixes:
        root = repo / prefix
        if root.is_file():
            rows.append({"path": prefix, "sha256": file_sha256(root)})
        elif root.is_dir():
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                if path.is_file() and "__pycache__" not in path.parts and "artifacts" not in path.parts:
                    rows.append({"path": path.relative_to(repo).as_posix(), "sha256": file_sha256(path)})
    return canonical_sha256(rows)


def _document_hashes(repo: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mechanism, paths in PROFILE_DOCUMENTS.items():
        result[mechanism] = {
            key: ({"path": path, "sha256": file_sha256(repo / path)} if key != "crosswalk_location" else path)
            for key, path in paths.items()
        }
    return result


def _dependency_lock_hashes(repo: Path) -> dict[str, str]:
    paths = (
        "pyproject.toml",
        "experiments/database_lineage/requirements.lock",
        "experiments/source_map_projection/requirements.lock",
        "experiments/source_map_projection/pnpm-lock.yaml",
        "experiments/opentelemetry_projection/requirements.lock",
        "experiments/w3c_prov_projection_v1/requirements.txt",
        "experiments/pytorch_autograd_training_lineage_v1/artifacts/pytorch_authority_manifest.json",
    )
    return {path: file_sha256(repo / path) for path in paths}


def _artifact_hashes(artifacts: Path) -> dict[str, str]:
    result = {}
    for path in sorted(artifacts.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "unified_manifest.json":
            result[path.relative_to(artifacts).as_posix()] = file_sha256(path)
    return result


def build_manifest(
    repo: Path,
    artifacts: Path,
    results: dict[str, dict[str, Any]],
    *,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    status_lines = _git(repo, "status", "--porcelain=v1").splitlines()
    core_changed = _core_changed_paths(repo)
    source_map_package = json.loads((repo / "experiments/source_map_projection/node_modules/source-map/package.json").read_text(encoding="utf-8"))
    wheel_authority = json.loads((repo / "experiments/pytorch_autograd_training_lineage_v1/artifacts/pytorch_authority_manifest.json").read_text(encoding="utf-8"))
    source_state_prefixes = (
        "experiments/five_profile_unified_projection_proof",
        "experiments/w3c_prov_projection_v1",
        "experiments/pytorch_autograd_training_lineage_v1",
    )
    return {
        "schema_version": "five-profile-unified-manifest-v1",
        "repository_commit": _git(repo, "rev-parse", "HEAD"),
        "repository_branch": _git(repo, "branch", "--show-current"),
        "repository_source_state_sha256": _tree_file_hash(repo, source_state_prefixes),
        "core_commit": CORE_COMMIT,
        "core_worktree_sha256": _tree_file_hash(repo, PROTECTED_PREFIXES),
        "core_changed_files": len(core_changed),
        "core_changed_paths": core_changed,
        "experiment_source_commits": {
            **SOURCE_COMMITS,
            "three_profile_unified_proof_v2": "7320fe8a2d690fc87da77d0739b432ea1812d63b",
            "w3c_upstream_equivalent_hardening_commit": "fbef9c23349f11075e1963ec867fa9f0fd767a35",
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "pytorch_version": _version("torch"),
            "pytorch_wheel": wheel_authority["wheel"],
            "node_version": _node_version(repo),
            "source_map_package_version": source_map_package["version"],
            "opentelemetry_api_version": _version("opentelemetry-api"),
            "opentelemetry_sdk_version": _version("opentelemetry-sdk"),
            "rdflib_version": _version("rdflib"),
            "duckdb_version": _version("duckdb"),
            "operating_system": platform.platform(),
        },
        "dependency_lock_hashes": _dependency_lock_hashes(repo),
        "frozen_profile_and_crosswalk_hashes": _document_hashes(repo),
        "runner_entrypoints": ENTRYPOINTS,
        "unified_entrypoint": "python -m experiments.five_profile_unified_projection_proof.scripts.run_all",
        "run_started_at_utc": started_at,
        "run_ended_at_utc": ended_at,
        "clean_worktree": not status_lines,
        "worktree_status_at_run": status_lines,
        "core_modified": bool(core_changed),
        "mechanism_final_statuses": {name: result["run_status"] for name, result in results.items()},
        "artifact_hashes": _artifact_hashes(artifacts),
        "compatibility_shims": [
            {
                "path": "experiments/pytorch_autograd_training_lineage_v1/core_compatibility.py",
                "scope": "protected Core tree audit only",
                "scientific_projection_logic_modified": False,
                "equivalence_test": "tests/experiments/five_profile_unified_projection_proof/test_autograd_compatibility.py",
            }
        ],
        "baseline_reconciliations": [
            {
                "mechanism": "pytorch_autograd_dependency_profile",
                "kind": "protected Core baseline reconciliation; no runtime shim",
                "historical_proof_core": "pre-6b3490 protected tree pinned by the imported experiment",
                "unified_core": CORE_COMMIT,
                "allowed_effect": "five enumerated snapshot/content-ID-bearing v1 baselines are regenerated against 6b3490; current graph P1, P2, queries, checkpoint semantics, and 29-relation hardening must remain exact",
                "core_files_changed_by_unified_package": len(core_changed),
            }
        ],
    }
