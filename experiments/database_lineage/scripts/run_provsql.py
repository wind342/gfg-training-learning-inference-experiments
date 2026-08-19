from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import duckdb
import psycopg

from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.provsql_reference import (
    compare_core_and_provsql,
    connect,
    execute_which,
    initialize_database,
    read_core_lineage,
    server_capabilities,
)
from experiments.database_lineage.src.tpch_loader import (
    official_sql_and_answers,
    scale_name,
)


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = REPO / "experiments" / "database_lineage"
RUNTIME = EXPERIMENT / "runtime"
ARTIFACT = EXPERIMENT / "artifacts" / "provsql_comparison.json"
REQUIRED = ((0.01, 1), (0.01, 3), (0.01, 6), (0.01, 10), (0.1, 1), (0.1, 6))
PINNED_IMAGE_DIGEST = "inriavalda/provsql@sha256:57c7877fe86638f201bc26fc0cb8ef759aeb09e9bfc03789c2d3a2b315305268"
PINNED_SOURCE_AUDIT = {
    "originally_requested_release": "v1.3.0",
    "selected_release": "v1.4.0",
    "git_commit": "37fc44474b75d3d0594e44b794b744675457eb7d",
    "source_url": "https://github.com/PierreSenellart/provsql/tree/v1.4.0",
    "release_notes_url": "https://provsql.org/releases/#version-1-4-0",
    "sr_which_available": True,
    "sr_how_available": True,
    "sr_why_available": True,
    "sr_counting_available": True,
    "sr_which_introduced_in": "1.4.0",
    "docker_hub_url": "https://hub.docker.com/layers/inriavalda/provsql/1.4.0/images/sha256-a5e3326de148f1a021df8eec2ae9f71b1f5bf672dd82bbaa9652791e6dcfe09e",
    "image_index_digest": "sha256:57c7877fe86638f201bc26fc0cb8ef759aeb09e9bfc03789c2d3a2b315305268",
    "linux_amd64_manifest_digest": "sha256:a5e3326de148f1a021df8eec2ae9f71b1f5bf672dd82bbaa9652791e6dcfe09e",
    "postgresql_major": 17,
    "build_environment": "debian:bookworm",
    "compile_commands": ["make -j$(nproc)", "make install"],
    "audit_note": "The originally specified ProvSQL 1.3.0 did not provide sr_which. ProvSQL 1.4.0 is the first formal release whose release notes and tagged source include the compiled sr_which evaluator.",
}


def image_digest() -> str | None:
    docker = os.environ.get("DOCKER", "docker")
    try:
        value = subprocess.run(
            [
                docker,
                "image",
                "inspect",
                "inriavalda/provsql:1.4.0",
                "--format",
                "{{index .RepoDigests 0}}",
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return PINNED_IMAGE_DIGEST
    return value or PINNED_IMAGE_DIGEST


def blocked(
    reason: str, *, capabilities: dict | None = None, error: str | None = None
) -> int:
    result = {
        "status": "external_comparator_unavailable",
        "required_image": f"inriavalda/provsql:1.4.0@{PINNED_IMAGE_DIGEST.rsplit('@', 1)[1]}",
        "image_digest": image_digest(),
        "reason": reason,
        "error": error,
        "capabilities": capabilities,
        "pinned_source_audit": PINNED_SOURCE_AUDIT,
        "required_evaluator": "sr_which(provenance(), 'provsql_tuple_mapping')",
        "which_lineage_exact_matches": 0,
        "total_output_rows": 0,
        "false_positives": None,
        "false_negatives": None,
        "supplementary_why_results": {"status": "not_run"},
        "supplementary_how_results": {"status": "not_run"},
        "supplementary_counting_results": {"status": "not_run"},
        "queries": {},
    }
    write_json(ARTIFACT, result)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "PROVSQL_DSN", "postgresql://test@127.0.0.1:5433/test?connect_timeout=5"
        ),
    )
    args = parser.parse_args()
    try:
        connection = connect(args.dsn)
    except psycopg.Error as exc:
        return blocked(
            "PostgreSQL/ProvSQL container is unreachable; no external lineage comparison was executed.",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS provsql CASCADE")
            cursor.execute("SET search_path TO public, provsql")
        capabilities = server_capabilities(connection)
        if capabilities["provsql_extension_version"] != "1.4.0":
            return blocked(
                "The connected ProvSQL extension is not the pinned 1.4.0 version.",
                capabilities=capabilities,
            )
        if not capabilities["sr_which_available"]:
            return blocked(
                "Pinned ProvSQL 1.4.0 unexpectedly does not expose sr_which, the mandatory independent which-lineage evaluator.",
                capabilities=capabilities,
            )
        results = {}
        total_exact = total_rows = false_positives = false_negatives = 0
        loaded_scales: set[float] = set()
        for scale, query in REQUIRED:
            name = scale_name(scale)
            if scale not in loaded_scales:
                initialize_database(connection, RUNTIME / f"tpch_sf_{name}_csv")
                loaded_scales.add(scale)
            db = duckdb.connect(str(RUNTIME / f"tpch_sf_{name}.duckdb"), read_only=True)
            db.execute("LOAD tpch")
            sql = official_sql_and_answers(db)["queries"][query]
            db.close()
            core_path = RUNTIME / f"core_lineage_sf_{name}_q{query}.json"
            core = read_core_lineage(core_path)
            provsql, elapsed = execute_which(connection, query, sql)
            comparison = compare_core_and_provsql(core, provsql)
            comparison["provsql_query_and_sr_which_seconds"] = elapsed
            results[f"sf_{name}_q{query}"] = comparison
            total_exact += comparison["exact_output_rows"]
            total_rows += comparison["output_rows"]
            false_positives += comparison["false_positives"]
            false_negatives += comparison["false_negatives"]
        artifact = {
            "status": "passed"
            if total_exact == total_rows and not false_positives and not false_negatives
            else "failed",
            "required_image": "inriavalda/provsql:1.4.0",
            "image_digest": image_digest(),
            "capabilities": capabilities,
            "pinned_source_audit": PINNED_SOURCE_AUDIT,
            "which_lineage_exact_matches": total_exact,
            "total_output_rows": total_rows,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "supplementary_why_results": {
                "status": "not_run",
                "available": capabilities["sr_why_available"],
            },
            "supplementary_how_results": {
                "status": "not_run",
                "available": capabilities["sr_how_available"],
            },
            "supplementary_counting_results": {
                "status": "not_run",
                "available": capabilities["sr_counting_available"],
            },
            "queries": results,
        }
        write_json(ARTIFACT, artifact)
        return 0 if artifact["status"] == "passed" else 1
    except (psycopg.Error, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return blocked(
            "ProvSQL evaluation failed before all mandatory comparisons completed.",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
