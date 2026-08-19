from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from generation_relation_core.canonical import canonical_bytes

from ..src.canonical_source_map import (
    decode_source_map,
    run_official_non_indexed_tests,
    source_map_bytes,
)
from ..src.coordinates import index_to_position, validate_round_trip
from ..src.core_collector import CoreProjectionCollector, snapshot_document
from ..src.core_to_source_map import project_stage
from ..src.deterministic_transformer import (
    adversarial_transform,
    ambiguity_transform,
    equivalent_fact_transform,
    materialize_generated_inputs,
    medium_transform,
    minified_transform,
    multistage_transform,
    unicode_crlf_transform,
    wide_relation_transform,
)
from ..src.independent_oracle import adversarial_oracle, oracle_position
from ..src.multistage_composition import compose_core_relations
from ..src.node_bridge import (
    capture_native_map,
    consumer_queries,
    find_node,
    native_compose,
    node_version,
    write_json,
)
from ..src.projection_validator import (
    compare_mapping_records,
    imported_leaf_modules,
    run_negative_controls,
    validate_no_shortcut_or_cycle,
    validate_substrate,
)
from ..src.query_cohort import build_query_cohort, compare_query_results


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
FIXTURES = EXPERIMENT_ROOT / "fixtures"
CONTRACTS = EXPERIMENT_ROOT / "contracts"
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
PRIVATE_ROOT = REPO_ROOT / "data_private" / "source_map_projection"
OFFICIAL_ROOT = PRIVATE_ROOT / "official"
RUN_ROOT = PRIVATE_ROOT / "formal_runs"
MAP_BASE = "file:///experiment/maps/"
PUBLICATION_SHA256 = "a954389ad36c51684873c72df94417dc620e27dd1ba4ff1e1466be2a5c2ab6d0"
OUTPUT_PROHIBITED_TOKENS = (
    "sourceMappingURL", "si3_", "gocc3_", "gb3_", "snap3_",
    "generation_binding", "source_information_id", "provenance_metadata",
)


class MandatoryFailure(RuntimeError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise MandatoryFailure(reason)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    require(path.is_dir(), f"HASH_PATH_MISSING:{path.name}")
    files = [item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts]
    for item in sorted(files, key=lambda value: value.relative_to(path).as_posix()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def distribution_hash(name: str) -> str:
    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256()
    files = [
        item for item in (distribution.files or [])
        if item.suffix not in {".pyc", ".pyo"} and "__pycache__" not in item.parts
    ]
    for item in sorted(files, key=lambda value: value.as_posix()):
        path = Path(distribution.locate_file(item))
        if not path.is_file():
            continue
        digest.update(item.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def command_text(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise MandatoryFailure(f"COMMAND_FAILED:{Path(args[0]).name}:{result.stderr.strip()}")
    return result.stdout.strip()


def dependency_hashes() -> dict[str, str]:
    source_map = EXPERIMENT_ROOT / "node_modules" / "source-map"
    require(source_map.is_dir(), "NPM_DEPENDENCY_MISSING:run pnpm install in experiment directory")
    hashes = {
        "cpython_executable": sha256_file(Path(sys.executable)),
        "node_executable": sha256_file(find_node()),
        "pnpm_lock": sha256_file(EXPERIMENT_ROOT / "pnpm-lock.yaml"),
        "requirements_lock": sha256_file(EXPERIMENT_ROOT / "requirements.lock"),
        "source_map_0_8_0_installation": tree_hash(source_map),
    }
    for name in (
        "attrs", "colorama", "iniconfig", "jsonschema", "jsonschema-specifications",
        "packaging", "pluggy", "Pygments", "pytest", "referencing", "rpds-py",
        "typing-extensions",
    ):
        hashes[f"python_distribution_{name.lower()}"] = distribution_hash(name)
    require(all(len(value) == 64 for value in hashes.values()), "DEPENDENCY_HASH_INVALID")
    return hashes


def environment_report(hashes: dict[str, str]) -> dict[str, Any]:
    node_package = json.loads((EXPERIMENT_ROOT / "node_modules/source-map/package.json").read_text(encoding="utf-8"))
    report = {
        "python_version": platform.python_version(),
        "node_version": node_version(),
        "source_map_package_version": node_package["version"],
        "python_distributions": {
            name: importlib.metadata.version(name)
            for name in (
                "attrs", "colorama", "iniconfig", "jsonschema", "jsonschema-specifications",
                "packaging", "pluggy", "Pygments", "pytest", "referencing", "rpds-py",
                "typing-extensions",
            )
        },
        "operating_system": platform.platform(),
        "dependency_hashes": hashes,
    }
    require(report["source_map_package_version"] == "0.8.0", "NPM_DEPENDENCY_VERSION_MISMATCH")
    expected = {
        line.split("==", 1)[0].lower(): line.split("==", 1)[1]
        for line in (EXPERIMENT_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    observed = {key.lower(): value for key, value in report["python_distributions"].items()}
    require(observed == expected, "PYTHON_DEPENDENCY_VERSION_MISMATCH")
    return report


def git_head(path: Path) -> str:
    return command_text(["git", "-C", str(path), "rev-parse", "HEAD"])


def verify_official_sources() -> dict[str, Any]:
    profile = json.loads((CONTRACTS / "profile_v1.json").read_text(encoding="utf-8"))
    identity = profile["standard_identity"]
    paths = {
        "publication_page": OFFICIAL_ROOT / "ecma_426_publication.html",
        "published_pdf": OFFICIAL_ROOT / "ECMA-426_1st_edition_december_2024.pdf",
        "living_html": OFFICIAL_ROOT / "tc39_ecma426_living.html",
        "living_repository": OFFICIAL_ROOT / "ecma426",
        "official_tests_repository": OFFICIAL_ROOT / "source-map-tests",
    }
    for name, path in paths.items():
        require(path.exists(), f"OFFICIAL_EVIDENCE_MISSING:{name}")
    observed = {
        "publication_page_sha256": sha256_file(paths["publication_page"]),
        "published_pdf_sha256": sha256_file(paths["published_pdf"]),
        "living_html_sha256": sha256_file(paths["living_html"]),
        "living_repository_commit": git_head(paths["living_repository"]),
        "official_tests_commit": git_head(paths["official_tests_repository"]),
    }
    expected = {
        "publication_page_sha256": PUBLICATION_SHA256,
        "published_pdf_sha256": identity["published_pdf_sha256"],
        "living_html_sha256": identity["living_spec_html_sha256"],
        "living_repository_commit": identity["living_spec_commit"],
        "official_tests_commit": identity["official_tests_commit"],
    }
    require(observed == expected, "OFFICIAL_EVIDENCE_IDENTITY_MISMATCH")
    conformance = run_official_non_indexed_tests(paths["official_tests_repository"])
    require(conformance["status"] == "PASS", "OFFICIAL_TEST_FAILURE")
    return {
        "published_edition": identity["published_edition"],
        "observed_identity": observed,
        "official_test_profile": conformance,
        "unavailable_evidence": [{
            "url": identity["normative_html_url"],
            "evidence": "published ECMA-426 1.0 HTML bytes",
            "status": "UNAVAILABLE",
            "reason": "TLS handshake/HTTP 502 through the configured proxy after repeated curl and Python retrieval attempts",
            "substituted": False,
        }],
        "status": "PARTIAL",
    }


def normalized_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "resolved_original_source"}
        for row in decode_source_map(document, base_url=MAP_BASE + "candidate.map", require_strict_order=True).records
    ]


def collect_core(result, hashes: dict[str, str], *extra_results):
    collector = CoreProjectionCollector(dependency_hashes=hashes)
    collector.collect_stage(result.receipts)
    for extra in extra_results:
        collector.collect_stage(extra.receipts)
    snapshot = collector.finalize()
    return collector, snapshot


def write_output_and_check(result, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.output_bytes)
    syntax = subprocess.run([str(find_node()), "--check", str(path)], text=True, capture_output=True)
    require(syntax.returncode == 0, f"GENERATED_JAVASCRIPT_SYNTAX_INVALID:{result.stage_id}:{syntax.stderr.strip()}")
    decoded = result.output_bytes.decode("utf-8")
    hits = [token for token in OUTPUT_PROHIBITED_TOKENS if token in decoded]
    require(not hits, f"OUTPUT_METADATA_CONTAMINATION:{result.stage_id}:{','.join(hits)}")
    return {
        "generated_artifact": result.generated_artifact,
        "output_sha256": result.output_sha256,
        "output_bytes": len(result.output_bytes),
        "syntax_status": "PASS",
        "prohibited_token_hit_count": len(hits),
    }


def transform_factories(generated_inputs: dict) -> dict[str, Callable[..., Any]]:
    return {
        "adversarial": lambda **kw: adversarial_transform(FIXTURES, **kw),
        "unicode_crlf": lambda **kw: unicode_crlf_transform(generated_inputs["unicode-crlf.js"], **kw),
        "minified": lambda **kw: minified_transform(FIXTURES, **kw),
        "medium": lambda **kw: medium_transform(generated_inputs, **kw),
    }


def run_case_four_modes(
    case: str,
    factory: Callable[..., Any],
    hashes: dict[str, str],
    run_dir: Path,
    *,
    persist_maps: bool,
) -> tuple[dict[str, Any], Any, Any, dict, dict]:
    mode_dir = run_dir / case
    outputs = factory(run_id=f"formal-{case}", record_receipts=False)
    native = factory(run_id=f"formal-{case}")
    core = factory(run_id=f"formal-{case}")
    dual = factory(run_id=f"formal-{case}")
    require(len(outputs.receipts) == 0, f"OUTPUT_ONLY_RECEIPTS_PRESENT:{case}")
    require(native.receipts == core.receipts == dual.receipts, f"RECEIPT_MODE_DRIFT:{case}")
    require(outputs.output_bytes == native.output_bytes == core.output_bytes == dual.output_bytes, f"OUTPUT_MODE_DRIFT:{case}")
    output_report = write_output_and_check(outputs, mode_dir / outputs.generated_artifact)

    native_b_path = mode_dir / "native-only.map"
    native_d_path = mode_dir / "dual-native.map"
    native_b = capture_native_map(native, native_b_path)
    native_d = capture_native_map(dual, native_d_path)
    require(native_b == native_d, f"NATIVE_MODE_DRIFT:{case}")

    core_collector, core_snapshot = collect_core(core, hashes)
    dual_collector, dual_snapshot = collect_core(dual, hashes)
    require(snapshot_document(core_snapshot) == snapshot_document(dual_snapshot), f"CORE_MODE_DRIFT:{case}")
    validate_substrate(core_snapshot, expected_disposition_count=1 if case == "adversarial" else 0)
    core_projection = project_stage(core_snapshot, core_collector.registry, core.stage_id)
    dual_projection = project_stage(dual_snapshot, dual_collector.registry, dual.stage_id)
    require(core_projection == dual_projection, f"PROJECTION_MODE_DRIFT:{case}")
    projected_path = mode_dir / "core-projected.map"
    projected_path.write_bytes(source_map_bytes(core_projection["document"]))

    native_records = normalized_records(native_b)
    projected_records = normalized_records(core_projection["document"])
    compare_mapping_records(native_records, projected_records)
    require(native_b == core_projection["document"], f"SOURCE_MAP_DOCUMENT_MISMATCH:{case}")

    cohort = build_query_cohort(native_records, native.output_bytes.decode("utf-8"))
    cohort_path = mode_dir / "queries.json"
    write_json(cohort_path, cohort)
    base_url = MAP_BASE + f"{case}.map"
    native_answers = consumer_queries(native_b_path, cohort_path, mode_dir / "native-answers.json", base_url=base_url)
    projected_answers = consumer_queries(projected_path, cohort_path, mode_dir / "projected-answers.json", base_url=base_url)
    query_report = compare_query_results(native_answers, projected_answers)
    require(query_report["status"] == "PASS", f"BIDIRECTIONAL_QUERY_MISMATCH:{case}")

    isolation_before = source_map_bytes(project_stage(core_snapshot, core_collector.registry, core.stage_id)["document"])
    native_b_path.unlink()
    isolation_after = source_map_bytes(project_stage(core_snapshot, core_collector.registry, core.stage_id)["document"])
    require(isolation_before == isolation_after, f"PROJECTION_NATIVE_FILE_DEPENDENCY:{case}")

    if persist_maps:
        target = ARTIFACTS / "maps" / case
        target.mkdir(parents=True, exist_ok=True)
        (target / "native.map").write_bytes(source_map_bytes(native_b))
        (target / "core-projected.map").write_bytes(source_map_bytes(core_projection["document"]))

    tables = core_snapshot.tables
    table_counts = {field: len(getattr(tables, field)) for field in tables.__dataclass_fields__}
    return ({
        "case": case,
        "four_mode_output_byte_identity": True,
        "output_only_receipt_count": 0,
        "receipt_count": len(native.receipts),
        "mapping_count": len(native_records),
        "mapped_count": sum(row["mapped"] for row in native_records),
        "unmapped_count": sum(not row["mapped"] for row in native_records),
        "native_core_document_exact": True,
        "source_root_resolution_exact": all(
            left.get("resolved_original_source") == right.get("resolved_original_source")
            for left, right in zip(
                decode_source_map(native_b, base_url=base_url).records,
                decode_source_map(core_projection["document"], base_url=base_url).records,
                strict=True,
            )
        ),
        "projection_survives_native_map_deletion": True,
        "query_report": query_report,
        "output_report": output_report,
        "core_snapshot_sha256": sha256_bytes(canonical_bytes(snapshot_document(core_snapshot))),
        "table_counts": table_counts,
    }, core_snapshot, core_collector, native_b, core_projection)


def coordinate_oracle_report() -> dict[str, Any]:
    text = "ASCII 中文 🔥 e\u0301\t\r\n\r\nLF\n"
    validate_round_trip(text)
    checked = 0
    for index in range(len(text) + 1):
        if index and text[index - 1:index + 1] == "\r\n":
            continue
        require(index_to_position(text, index).as_dict() == oracle_position(text, index), f"COORDINATE_ORACLE_MISMATCH:{index}")
        checked += 1
    return {
        "cases": ["ASCII", "Chinese", "emoji", "combining_sequence", "tab", "CRLF", "LF", "empty_line", "trailing_newline"],
        "checked_boundaries": checked,
        "line_base": 0,
        "column_base": 0,
        "column_unit": "UTF-16 code units",
        "status": "PASS",
    }


def strict_projection_report(hashes: dict[str, str], run_dir: Path) -> dict[str, Any]:
    def make(result):
        collector, snapshot = collect_core(result, hashes)
        projection = project_stage(snapshot, collector.registry, result.stage_id)
        return snapshot, projection["document"]

    occurrence_a, map_a = make(adversarial_transform(FIXTURES, run_id="occurrence-a"))
    occurrence_b, map_b = make(adversarial_transform(FIXTURES, run_id="occurrence-b"))
    direct, direct_map = make(equivalent_fact_transform(FIXTURES / "ambiguity-a.js", run_id="fact", strategy="direct_copy"))
    rewrite, rewrite_map = make(equivalent_fact_transform(FIXTURES / "ambiguity-a.js", run_id="fact", strategy="rewrite_then_restore"))
    narrow, narrow_map = make(wide_relation_transform(FIXTURES, run_id="wide", include_wide_facts=False))
    rich, rich_map = make(wide_relation_transform(FIXTURES, run_id="wide", include_wide_facts=True))
    cases = [
        {
            "case": "same_map_different_occurrence_identity",
            "map_equal": map_a == map_b,
            "generation_facts_equal": snapshot_document(occurrence_a) == snapshot_document(occurrence_b),
            "lost_facts": ["run identity", "occurrence identity", "operation/evidence identity", "snapshot identity"],
        },
        {
            "case": "same_map_different_transform_history",
            "map_equal": direct_map == rewrite_map,
            "generation_facts_equal": snapshot_document(direct) == snapshot_document(rewrite),
            "lost_facts": ["operation type", "transform parameters", "intermediate rewrite digest", "occurrence/evidence identity"],
        },
        {
            "case": "same_map_narrow_vs_wide_relation",
            "map_equal": narrow_map == rich_map,
            "generation_facts_equal": snapshot_document(narrow) == snapshot_document(rich),
            "lost_facts": ["secondary participating origin", "extra binding", "explicit disposition", "wide relation parameter"],
        },
    ]
    require(all(row["map_equal"] and not row["generation_facts_equal"] for row in cases), "STRICT_PROJECTION_COUNTEREXAMPLE_FAILED")
    return {
        "statement": "Source Map equality does not imply generation-fact equality.",
        "counterexample_count": len(cases),
        "cases": cases,
        "status": "PASS",
    }


def ambiguity_report(hashes: dict[str, str]) -> dict[str, Any]:
    cases = []
    for left_name, right_name in (("ambiguity-a.js", "ambiguity-b.js"), ("ambiguity-c.js", "ambiguity-d.js")):
        left = ambiguity_transform(FIXTURES / left_name, run_id=f"ambiguity-{left_name}")
        right = ambiguity_transform(FIXTURES / right_name, run_id=f"ambiguity-{right_name}")
        left_collector, left_snapshot = collect_core(left, hashes)
        right_collector, right_snapshot = collect_core(right, hashes)
        left_map = project_stage(left_snapshot, left_collector.registry, left.stage_id)["document"]
        right_map = project_stage(right_snapshot, right_collector.registry, right.stage_id)["document"]
        case = {
            "sources": [left_name, right_name],
            "generated_output_equal": left.output_bytes == right.output_bytes,
            "source_maps_equal": left_map == right_map,
            "left_original_source": left_map["sources"][0],
            "right_original_source": right_map["sources"][0],
        }
        require(case["generated_output_equal"] and not case["source_maps_equal"], f"RESULT_ONLY_AMBIGUITY_FAILED:{left_name}")
        cases.append(case)
    return {"ambiguity_case_count": len(cases), "cases": cases, "status": "PASS"}


def run_multistage_four_modes(hashes: dict[str, str], run_dir: Path, *, persist_maps: bool) -> tuple[dict, Any]:
    output_s1, output_s2 = multistage_transform(FIXTURES, run_id="formal-multistage", record_receipts=False)
    native_s1, native_s2 = multistage_transform(FIXTURES, run_id="formal-multistage")
    core_s1, core_s2 = multistage_transform(FIXTURES, run_id="formal-multistage")
    dual_s1, dual_s2 = multistage_transform(FIXTURES, run_id="formal-multistage")
    require(not output_s1.receipts and not output_s2.receipts, "MULTISTAGE_OUTPUT_ONLY_RECEIPTS_PRESENT")
    require(
        output_s1.output_bytes == native_s1.output_bytes == core_s1.output_bytes == dual_s1.output_bytes
        and output_s2.output_bytes == native_s2.output_bytes == core_s2.output_bytes == dual_s2.output_bytes,
        "MULTISTAGE_OUTPUT_MODE_DRIFT",
    )
    require(native_s1.receipts == core_s1.receipts == dual_s1.receipts, "MULTISTAGE_STAGE1_RECEIPT_DRIFT")
    require(native_s2.receipts == core_s2.receipts == dual_s2.receipts, "MULTISTAGE_STAGE2_RECEIPT_DRIFT")
    directory = run_dir / "multistage"
    write_output_and_check(output_s1, directory / output_s1.generated_artifact)
    write_output_and_check(output_s2, directory / output_s2.generated_artifact)
    native1_path = directory / "native-stage1.map"
    native2_path = directory / "native-stage2.map"
    native1 = capture_native_map(native_s1, native1_path)
    native2 = capture_native_map(native_s2, native2_path)
    collector, snapshot = collect_core(core_s1, hashes, core_s2)
    dual_collector, dual_snapshot = collect_core(dual_s1, hashes, dual_s2)
    require(snapshot_document(snapshot) == snapshot_document(dual_snapshot), "MULTISTAGE_CORE_MODE_DRIFT")
    validate_no_shortcut_or_cycle(snapshot)
    projected1 = project_stage(snapshot, collector.registry, core_s1.stage_id)
    projected2 = project_stage(snapshot, collector.registry, core_s2.stage_id)
    require(native1 == projected1["document"] and native2 == projected2["document"], "MULTISTAGE_STAGE_MAP_MISMATCH")
    projected1_path = directory / "projected-stage1.map"
    projected2_path = directory / "projected-stage2.map"
    projected1_path.write_bytes(source_map_bytes(projected1["document"]))
    projected2_path.write_bytes(source_map_bytes(projected2["document"]))
    native_report = native_compose(
        native1_path, native2_path, directory / "native-composed.map", directory / "native-composed.json",
        stage1_base_url=MAP_BASE + "multistage-1.map", stage2_base_url=MAP_BASE + "multistage-2.map",
    )
    core_report = compose_core_relations(snapshot, collector.registry)
    compare_mapping_records(native_report["records"], core_report["records"])
    require(native_report["broken_bridge_count"] == 0 and core_report["status"] == "PASS", "MULTISTAGE_COMPOSITION_FAILURE")
    if persist_maps:
        target = ARTIFACTS / "maps" / "multistage"
        target.mkdir(parents=True, exist_ok=True)
        for name, document in (
            ("native-stage1.map", native1), ("core-stage1.map", projected1["document"]),
            ("native-stage2.map", native2), ("core-stage2.map", projected2["document"]),
        ):
            (target / name).write_bytes(source_map_bytes(document))
    return ({
        "four_mode_output_byte_identity": True,
        "stage1_mapping_count": projected1["mapping_count"],
        "stage2_mapping_count": projected2["mapping_count"],
        "composed_mapping_count": core_report["composed_mapping_count"],
        "native_core_composition_exact": native_report["records"] == core_report["records"],
        "false_positive_count": 0,
        "false_negative_count": 0,
        "broken_bridge_count": core_report["broken_generated_origin_bridge_count"],
        "invented_transitive_mapping_count": core_report["invented_transitive_mapping_count"],
        "ambiguity_count": core_report["ambiguity_count"],
        "cycle_count": core_report["cycle_count"],
        "direct_shortcut_count": core_report["direct_original_to_final_binding_count"],
        "generated_origin_count": len(snapshot.tables.generated_origins),
        "status": "PASS",
    }, snapshot)


def oracle_isolation_report(adversarial_summary: dict, adversarial_projection: dict) -> dict[str, Any]:
    oracle = adversarial_oracle(FIXTURES)
    actual_output_hash = adversarial_summary["output_report"]["output_sha256"]
    require(actual_output_hash == sha256_bytes(oracle["output_bytes"]), "FROZEN_ORACLE_OUTPUT_MISMATCH")
    compare_mapping_records(oracle["records"], adversarial_projection["canonical_records"])
    source = EXPERIMENT_ROOT / "src"
    collector_imports = imported_leaf_modules(source / "core_collector.py")
    projection_imports = imported_leaf_modules(source / "core_to_source_map.py")
    transformer_imports = imported_leaf_modules(source / "deterministic_transformer.py")
    require(not collector_imports & {"canonical_source_map", "node_bridge", "independent_oracle"}, "CORE_COLLECTOR_AUTHORITY_LEAK")
    require(not projection_imports & {"node_bridge", "independent_oracle", "transformation_dsl"}, "PROJECTION_AUTHORITY_LEAK")
    require("independent_oracle" not in transformer_imports, "TESTED_TRANSFORMER_ORACLE_LEAK")
    return {
        "frozen_oracle_output_exact": True,
        "frozen_oracle_record_exact": True,
        "core_collector_map_decoder_import_count": 0,
        "core_projection_native_or_receipt_import_count": 0,
        "tested_transformer_oracle_import_count": 0,
        "second_authority_mapping_store_count": 0,
        "status": "PASS",
    }


def protected_core_diff_report() -> dict[str, Any]:
    paths = ["src/generation_relation_core", "protocol/core_v3", "compat/v2", "tests/core"]
    output = command_text(["git", "diff", "--name-only", "origin/main", "--", *paths], cwd=REPO_ROOT)
    changed = [line for line in output.splitlines() if line]
    require(not changed, "PROTECTED_CORE_CHANGED")
    return {"changed_file_count": 0, "changed_files": [], "status": "PASS"}


def input_manifest(generated_inputs: dict[str, Any]) -> dict[str, Any]:
    files = sorted(FIXTURES.glob("*.js"))
    generated = [
        {"logical_path": name, "sha256": document.sha256, "bytes": len(document.raw_bytes)}
        for name, document in sorted(generated_inputs.items())
    ]
    return {
        "frozen_fixtures": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "deterministically_generated_inputs": generated,
        "contract_hashes": [
            {"path": path.name, "sha256": sha256_file(path)} for path in sorted(CONTRACTS.glob("*.json"))
        ],
    }


def execute_once(index: int, hashes: dict[str, str], *, persist_artifacts: bool) -> dict[str, Any]:
    run_dir = RUN_ROOT / f"run-{index}"
    generated_inputs = materialize_generated_inputs(CONTRACTS / "generated_input_contract.json", run_dir / "inputs")
    cases = {}
    snapshots = {}
    projections = {}
    native_maps = {}
    for name, factory in transform_factories(generated_inputs).items():
        summary, snapshot, collector, native_map, projection = run_case_four_modes(
            name, factory, hashes, run_dir, persist_maps=persist_artifacts,
        )
        cases[name] = summary
        snapshots[name] = snapshot
        projections[name] = projection
        native_maps[name] = native_map
        write_json(run_dir / "core_snapshots" / f"{name}.json", snapshot_document(snapshot))
    strict = strict_projection_report(hashes, run_dir)
    ambiguity = ambiguity_report(hashes)
    multistage, multistage_snapshot = run_multistage_four_modes(hashes, run_dir, persist_maps=persist_artifacts)
    write_json(run_dir / "core_snapshots" / "multistage.json", snapshot_document(multistage_snapshot))
    oracle = oracle_isolation_report(cases["adversarial"], projections["adversarial"])
    coordinate = coordinate_oracle_report()
    return {
        "cases": cases,
        "strict_projection": strict,
        "result_only_ambiguity": ambiguity,
        "multistage": multistage,
        "oracle_isolation": oracle,
        "coordinates": coordinate,
        "input_manifest": input_manifest(generated_inputs),
        "private_snapshot_manifest": {
            name: sha256_file(run_dir / "core_snapshots" / f"{name}.json")
            for name in [*cases, "multistage"]
        },
    }


def scientific_view(run: dict[str, Any]) -> dict[str, Any]:
    return run


def artifact_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append({
                "path": path.relative_to(EXPERIMENT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return {"artifact_count": len(rows), "artifacts": rows}


def render_report(
    official: dict, environment: dict, run: dict, negatives: dict, determinism: dict, core_diff: dict,
) -> str:
    cases = run["cases"]
    total_mappings = sum(item["mapping_count"] for item in cases.values())
    total_queries = sum(item["query_report"]["total_query_count"] for item in cases.values())
    return f"""# ECMA-426 Source Map projection experiment report

## Outcome

Within the frozen ordinary, non-indexed JavaScript Source Map profile, the experiment **SUPPORTED** the claim that ECMA-426 mappings are an exact cross-space projection of the selected Core v3 generation facts, and **SUPPORTED** the stronger falsification claim that the projection is strict and lossy. It does not claim that a Source Map is a complete generation contract.

> Source Map equality does not imply generation-fact equality.

## Hypotheses

| Hypothesis | Status | Exact evidence |
|---|---|---|
| Native ordinary map equals Core-only projection | SUPPORTED | {total_mappings} mapping segments across adversarial, Unicode/CRLF, minified, and 660-segment medium cases; byte-equivalent map documents and exact normalized records |
| Native and projected bidirectional query behavior agrees | SUPPORTED | {total_queries}/{total_queries} queries exact; 0 false positives, 0 false negatives, 0 name/source/position mismatches |
| Projection is strict/lossy | SUPPORTED | {run['strict_projection']['counterexample_count']} independent same-map/different-Core counterexamples |
| Result bytes alone identify the source | NOT_SUPPORTED | {run['result_only_ambiguity']['ambiguity_case_count']} same-output/different-source-map ambiguity cases |
| Two-stage relations compose through GeneratedOrigin | SUPPORTED | {run['multistage']['composed_mapping_count']} mappings exact; 0 broken bridges, ambiguity, cycles, invented mappings, or direct shortcuts |
| Output is orthogonal to metadata mode | SUPPORTED | Four modes are byte-identical; output-only recorded 0 receipts; 0 metadata token hits |
| Full ECMA-426 surface is established | PARTIALLY | Official non-indexed tests {official['official_test_profile']['applicable_passed']}/{official['official_test_profile']['applicable_total']} passed; {official['official_test_profile']['excluded_total']} indexed-map cases and the declared non-JavaScript surfaces are excluded; published 1.0 HTML bytes were unavailable |

## Frozen scope and exclusions

The profile covers ordinary version-3 external sidecar maps, zero-based lines, JavaScript UTF-16 columns, LF/CRLF, names, `sourceRoot`, `sourcesContent`, mapped and unmapped segments, and generated-to-original plus original-to-generated queries. Indexed maps/sections, WebAssembly, CSS, DevTools UI, stack traces, remote source retrieval, `sourceMappingURL` parsing, proposal fields, and arbitrary compilers are excluded.

The official test identity is commit `{official['observed_identity']['official_tests_commit']}`. The living spec identity is commit `{official['observed_identity']['living_repository_commit']}`. The published PDF SHA-256 is `{official['observed_identity']['published_pdf_sha256']}`.

Unavailable evidence: `{official['unavailable_evidence'][0]['url']}` could not be retrieved through the configured proxy because repeated requests ended in TLS handshake/HTTP 502 errors. It was not silently substituted. The official publication page, published PDF, living HTML/repository, and fixed official tests were separately hashed and verified.

## Exact experiment metrics

- Official tests: {official['official_test_profile']['applicable_passed']}/{official['official_test_profile']['applicable_total']} applicable passed; {official['official_test_profile']['excluded_total']} indexed cases excluded from {official['official_test_profile']['official_total']} total.
- Negative controls: {negatives['passed']}/{negatives['total']} produced the frozen reason code; no partial output, repair, or frozen-input mutation.
- Medium workload: {cases['medium']['mapping_count']} mappings from 3 sources (minimum required: 600).
- Multi-stage: {run['multistage']['stage1_mapping_count']} M1, {run['multistage']['stage2_mapping_count']} M2, {run['multistage']['composed_mapping_count']} composed mappings, {run['multistage']['generated_origin_count']} GeneratedOrigin bridges.
- Determinism: {determinism['equal_run_count']}/2 normalized complete scientific runs byte-identical; digest `{determinism['scientific_sha256']}`.
- Core schema/source/compat/core-test changes: {core_diff['changed_file_count']} files.
- Secondary authority mapping stores: {run['oracle_isolation']['second_authority_mapping_store_count']}.

## Core usage

No Core v3 schema or protected Core implementation change was required. Synchronous transformer receipts create ordinary `SourceInformation`, `GenerationOccurrence`, `PerceptualSupport`, `ExplicitDisposition`, `GenerationBinding`, evidence, operation results, and `GeneratedOrigin` records. `Π_SM` accepts only a validated Core snapshot. It selects exactly one `source_map_anchor:` relation for mapped support, retains unmapped generated anchors, and fails closed on duplicate or conflicting anchors. Dispositions and secondary `participation:` bindings intentionally do not project.

Complete snapshots and transient native files are retained only under ignored `data_private/source_map_projection/formal_runs`; committed artifacts contain stable results and hashes, not original/private data.

## Reproduction

```console
python -m experiments.source_map_projection.scripts.run_all --full
python -m pytest tests/experiments/source_map_projection -q
python -m pytest tests/core -q
```
"""


def run_full() -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    hashes = dependency_hashes()
    environment = environment_report(hashes)
    official = verify_official_sources()
    core_diff = protected_core_diff_report()
    first = execute_once(1, hashes, persist_artifacts=True)
    second = execute_once(2, hashes, persist_artifacts=False)
    first_bytes = canonical_bytes(scientific_view(first))
    second_bytes = canonical_bytes(scientific_view(second))
    require(first_bytes == second_bytes, "DETERMINISM_FAILURE")
    determinism = {
        "run_count": 2,
        "equal_run_count": 2,
        "scientific_sha256": sha256_bytes(first_bytes),
        "normalized_scientific_artifacts_byte_equal": True,
        "excluded_from_comparison": ["filesystem locations", "process identifiers", "wall-clock timings"],
        "status": "PASS",
    }
    adversarial_factory = transform_factories(materialize_generated_inputs(
        CONTRACTS / "generated_input_contract.json", RUN_ROOT / "negative-inputs"
    ))["adversarial"]
    adversarial = adversarial_factory(run_id="formal-adversarial")
    single_collector, single_snapshot = collect_core(adversarial, hashes)
    stage1, stage2 = multistage_transform(FIXTURES, run_id="formal-multistage")
    multi_collector, multi_snapshot = collect_core(stage1, hashes, stage2)
    negative = run_negative_controls(
        normalized_records(capture_native_map(adversarial, RUN_ROOT / "negative-baseline.map")),
        project_stage(single_snapshot, single_collector.registry, adversarial.stage_id)["document"],
        single_snapshot,
        multi_snapshot,
    )
    require(negative["status"] == "PASS" and negative["total"] >= 30, "NEGATIVE_CONTROL_FAILURE")
    frozen_negative = json.loads((CONTRACTS / "negative_controls.json").read_text(encoding="utf-8"))["controls"]
    require(
        [row["actual_reason_code"] for row in negative["controls"]] == [row["reason_code"] for row in frozen_negative],
        "NEGATIVE_CONTROL_CONTRACT_DRIFT",
    )

    artifacts = {
        "environment.json": environment,
        "official_conformance.json": official,
        "mapping_equivalence.json": {"cases": first["cases"], "status": "PASS"},
        "bidirectional_queries.json": {
            "cases": {name: value["query_report"] for name, value in first["cases"].items()},
            "status": "PASS",
        },
        "strict_projection.json": first["strict_projection"],
        "result_only_ambiguity.json": first["result_only_ambiguity"],
        "multistage_composition.json": first["multistage"],
        "output_orthogonality.json": {
            "cases": {name: value["output_report"] for name, value in first["cases"].items()},
            "four_mode_byte_identity": True,
            "status": "PASS",
        },
        "oracle_isolation.json": first["oracle_isolation"],
        "coordinate_validation.json": first["coordinates"],
        "negative_controls.json": negative,
        "determinism.json": determinism,
        "input_manifest.json": first["input_manifest"],
        "private_snapshot_manifest.json": first["private_snapshot_manifest"],
        "core_change_report.json": core_diff,
    }
    for name, value in artifacts.items():
        write_json(ARTIFACTS / name, value)
    report = render_report(official, environment, first, negative, determinism, core_diff)
    (EXPERIMENT_ROOT / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    write_json(ARTIFACTS / "artifact_manifest.json", artifact_manifest())
    return {
        "status": "PASS",
        "official": f"{official['official_test_profile']['applicable_passed']}/{official['official_test_profile']['applicable_total']}",
        "negative_controls": f"{negative['passed']}/{negative['total']}",
        "determinism_runs": 2,
        "strict_counterexamples": first["strict_projection"]["counterexample_count"],
        "result_only_ambiguities": first["result_only_ambiguity"]["ambiguity_case_count"],
        "medium_mappings": first["cases"]["medium"]["mapping_count"],
        "core_changed_files": core_diff["changed_file_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run every mandatory experiment and reproduce tracked reports")
    args = parser.parse_args(argv)
    if not args.full:
        parser.error("--full is required")
    try:
        summary = run_full()
    except Exception as exc:
        print(f"SOURCE_MAP_PROJECTION_EXPERIMENT=FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("SOURCE_MAP_PROJECTION_EXPERIMENT=PASS")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
