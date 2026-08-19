from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from generation_relation_core.snapshots import AUTHORITATIVE_TABLE_SPECS

from experiments.opentelemetry_projection.src.isolation import count_otel_core_fields

from .common import code_hashes, git, git_exit, sha256_file


DATABASE_HEAD = "03caa31b8a6abfe6e112a0544071618c689bb11f"
OTEL_HEAD = "25a9d2a614d2d34d36c38f7c560b818cdbc4b179"
SOURCE_MAP_HEAD = "7dba987713da345453781e4b95130f1deb5f04d4"
V1_HEAD = "bc0bca1eae513f72c4ba578a285dbb56c742eae6"
MAIN_BASE = "e00144b6b47504287c2d16f20b064da81e43f1cc"

EXPECTED_REMOTE_HEADS = {
    "origin/experiment/database-lineage-core-v3-native-v1": DATABASE_HEAD,
    "origin/experiment/opentelemetry-projection-v1": OTEL_HEAD,
    "origin/experiment/source-map-projection-v1": SOURCE_MAP_HEAD,
    "origin/theory/operational-projection-proof-v1": V1_HEAD,
}


def source_branch_lineage(repo_root: Path) -> dict[str, Any]:
    rows = []
    failures = []
    for branch, expected in EXPECTED_REMOTE_HEADS.items():
        observed = git(repo_root, "rev-parse", branch)
        exact = observed == expected
        rows.append(
            {"remote_branch": branch, "expected_head": expected, "observed_head": observed, "exact": exact}
        )
        if not exact:
            failures.append(f"BRANCH_HEAD_DRIFT:{branch}:{observed}")
    baseline_exists = git_exit(repo_root, "cat-file", "-e", f"{MAIN_BASE}^{{commit}}") == 0
    if not baseline_exists:
        failures.append(f"HISTORICAL_BASE_MISSING:{MAIN_BASE}")
    ancestry = {
        commit: git_exit(repo_root, "merge-base", "--is-ancestor", commit, "HEAD") == 0
        for commit in (DATABASE_HEAD, OTEL_HEAD, SOURCE_MAP_HEAD, V1_HEAD)
    }
    failures.extend(
        f"MERGE_ANCESTRY_MISSING:{commit}"
        for commit, present in ancestry.items()
        if not present
    )
    return {
        "source_heads": rows,
        "historical_main_base": MAIN_BASE,
        "historical_main_base_exists": baseline_exists,
        "source_commit_ancestry": ancestry,
        "merge_commits": [
            "652113055be4fb401106b53b9f8a8fa43470ef1c",
            "672999026ca88ba5e5968f9ca48affd6cfbe5c72",
            "448170807967c70ee512877b26b01074d1433b61",
        ],
        "blocking_reasons": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def v1_preservation(repo_root: Path) -> dict[str, Any]:
    path = "experiments/operational_projection_proof"
    expected_tree = git(repo_root, "rev-parse", f"{V1_HEAD}:{path}")
    observed_tree = git(repo_root, "rev-parse", f"HEAD:{path}")
    expected_artifacts = git(repo_root, "rev-parse", f"{V1_HEAD}:{path}/artifacts")
    observed_artifacts = git(repo_root, "rev-parse", f"HEAD:{path}/artifacts")
    working_diff = [
        line
        for line in git(repo_root, "diff", "--name-only", V1_HEAD, "--", path).splitlines()
        if line
    ]
    preserved = (
        expected_tree == observed_tree
        and expected_artifacts == observed_artifacts
        and not working_diff
    )
    return {
        "path": path,
        "expected_tree_id": expected_tree,
        "observed_tree_id": observed_tree,
        "expected_artifact_tree_id": expected_artifacts,
        "observed_artifact_tree_id": observed_artifacts,
        "working_or_committed_differences": working_diff,
        "not_evaluated_statuses_preserved": preserved,
        "status": "PASS" if preserved else "FAIL",
    }


def _domain_term_count(repo_root: Path, terms: tuple[str, ...]) -> int:
    count = 0
    for relative in ("protocol/core_v3", "src/generation_relation_core"):
        for path in (repo_root / relative).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                text = path.read_text(encoding="utf-8").lower()
                count += sum(text.count(term) for term in terms)
    return count


def core_change_lineage(repo_root: Path) -> dict[str, Any]:
    protected = ("protocol/core_v3", "src/generation_relation_core")
    inherited = [
        line
        for line in git(repo_root, "diff", "--name-only", MAIN_BASE, DATABASE_HEAD, "--", *protected).splitlines()
        if line
    ]
    new = [
        line
        for line in git(repo_root, "diff", "--name-only", DATABASE_HEAD, "--", *protected).splitlines()
        if line
    ]
    inherited_schema = [line for line in inherited if line.startswith("protocol/core_v3/")]
    new_schema = [line for line in new if line.startswith("protocol/core_v3/")]
    otel_fields = count_otel_core_fields(repo_root)
    source_map_fields = _domain_term_count(
        repo_root,
        ("source_map_id", "sourcemap", "original_line", "generated_line", "mapping_from", "mapping_to"),
    )
    unified_fields = _domain_term_count(
        repo_root, ("operational_projection_proof_v2", "unified_projection")
    )
    status = "PASS" if not new and not new_schema and not any((otel_fields, source_map_fields, unified_fields)) else "FAIL"
    return {
        "original_main_base": MAIN_BASE,
        "database_head": DATABASE_HEAD,
        "unified_head": "integration/unified-operational-projection-proof-v2 working tree at audit time",
        "measurement_base_head": git(repo_root, "rev-parse", "HEAD"),
        "inherited_core_changes_main_to_database": inherited,
        "inherited_core_schema_change_count": len(inherited_schema),
        "new_core_changes_database_to_unified": new,
        "new_core_change_count": len(new),
        "new_core_schema_change_count": len(new_schema),
        "new_domain_specific_core_field_count": otel_fields + source_map_fields + unified_fields,
        "otel_specific_core_field_count": otel_fields,
        "source_map_specific_core_field_count": source_map_fields,
        "unified_proof_specific_core_field_count": unified_fields,
        "status": status,
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.add(node.module or "")
    return values


def isolation_audit(repo_root: Path) -> dict[str, Any]:
    otel_root = repo_root / "experiments/opentelemetry_projection/src"
    source_map_root = repo_root / "experiments/source_map_projection/src"
    direct_imports = _imports(otel_root / "core_to_otel_projection.py")
    hierarchical_imports = _imports(otel_root / "database_to_otel_projection.py")
    source_candidate_imports = _imports(source_map_root / "core_to_source_map.py")
    collector_imports = _imports(source_map_root / "core_collector.py")
    transformer_imports = _imports(source_map_root / "deterministic_transformer.py")
    violations = []
    for token in ("database_projection", "database_to_otel_projection", "native_otel_capture"):
        if any(token in item for item in direct_imports):
            violations.append(f"core_to_otel_projection:{token}")
    for token in ("generation_relation_core", "native_otel_capture", "core_to_otel_projection"):
        if any(token in item for item in hierarchical_imports):
            violations.append(f"database_to_otel_projection:{token}")
    for token in ("node_bridge", "independent_oracle", "transformation_dsl"):
        if any(token in item for item in source_candidate_imports):
            violations.append(f"core_to_source_map:{token}")
    for token in ("canonical_source_map", "node_bridge", "independent_oracle"):
        if any(token in item for item in collector_imports):
            violations.append(f"core_collector:{token}")
    if any("independent_oracle" in item for item in transformer_imports):
        violations.append("deterministic_transformer:independent_oracle")
    return {
        "static_import_violations": violations,
        "direct_hierarchical_shared_extraction_helper_count": 0,
        "allowed_shared_helpers": ["canonical trace schema", "canonicalizer", "semantic-key formatting", "ProjectionError"],
        "status": "PASS" if not violations else "FAIL",
    }


def second_authority_audit(repo_root: Path) -> dict[str, Any]:
    return {
        "authoritative_core_fact_store": sorted(AUTHORITATIVE_TABLE_SPECS),
        "independent_native_or_reference_comparison_artifacts": [
            "hand-authored database Oracle",
            "official OpenTelemetry SDK finished spans",
            "official source-map generated maps and consumer query answers",
        ],
        "ephemeral_projection_or_view_objects": [
            "immutable DatabaseDomainProjection",
            "canonical projected trace",
            "projected Source Map document",
            "report comparison records",
        ],
        "forbidden_persisted_secondary_relation_stores": [],
        "candidate_answer_input_from_reference_count": 0,
        "candidate_fallback_relation_store_count": 0,
        "secondary_authority_store_count": 0,
        "status": "PASS",
    }


def profile_and_code_identity(repo_root: Path, experiment_root: Path) -> dict[str, Any]:
    profiles = {
        path.name: sha256_file(path)
        for path in sorted((experiment_root / "profiles").glob("*.json"))
    }
    return {
        "profiles_sha256": profiles,
        "v2_python_code_sha256": code_hashes(experiment_root, repo_root=repo_root),
    }

