from __future__ import annotations

import ast
from pathlib import Path

from experiments.w3c_prov_projection_v1.src.candidate_projection import project_snapshot
from experiments.w3c_prov_projection_v1.src.core_capture import CoreCaptureCollector
from experiments.w3c_prov_projection_v1.src.generator import run_generator
from experiments.w3c_prov_projection_v1.src.native_reference import NativeProvCollector
from experiments.w3c_prov_projection_v1.src.provn import parse_provn, serialize_provn
from experiments.w3c_prov_projection_v1.src.provo_normalizer import normalize_provo
from experiments.w3c_prov_projection_v1.src.record_model import validate_normalized_records


ROOT = Path(__file__).resolve().parents[1]


def _run_dual():
    core = CoreCaptureCollector()
    native = NativeProvCollector()
    output = run_generator([core, native])
    snapshot = core.validated_snapshot()
    candidate = project_snapshot(snapshot)
    provn = serialize_provn(candidate)
    ttl = native.qualified_provo()
    return output, snapshot, candidate, provn, ttl


def test_candidate_native_and_representations_are_exact() -> None:
    _output, _snapshot, candidate, provn, ttl = _run_dual()
    assert parse_provn(provn) == candidate
    assert normalize_provo(ttl) == candidate
    assert validate_normalized_records(candidate) == []


def test_fixture_has_required_real_structure() -> None:
    _output, snapshot, candidate, _provn, _ttl = _run_dual()
    tables = snapshot.tables
    assert len(tables.source_information_records) == 4
    values = [row["source_payload"]["value"] for row in tables.source_information_records]
    assert values.count({"key": "K1", "value": 1}) == 2
    assert len(tables.generation_occurrences) == 3
    assert len(tables.generated_origins) == 2
    assert len(tables.explicit_dispositions) == 1
    assert len(tables.generation_bindings) == 14
    assert len([row for row in candidate if row["kind"] == "usage"]) == 14
    assert len([row for row in candidate if row["kind"] == "derivation"]) == 14
    assert len([row for row in candidate if row["kind"] == "generation"]) == 6
    assert len([row for row in candidate if row["kind"] == "association"]) == 3


def test_stage_two_is_true_two_by_two_without_original_shortcuts() -> None:
    _output, snapshot, _candidate, _provn, _ttl = _run_dual()
    occurrences = {row["generation_occurrence_id"]: row for row in snapshot.tables.generation_occurrences}
    stage_two = {
        row["generation_occurrence_id"] for row in snapshot.tables.generation_occurrences
        if row["occurrence_stage"] == "stage-2"
    }
    stage_two_bindings = [
        row for row in snapshot.tables.generation_bindings
        if row["generation_occurrence_id"] in stage_two
    ]
    assert len(stage_two_bindings) == 4
    assert {row["origin_reference"]["kind"] for row in stage_two_bindings} == {"generated_origin"}
    assert len({row["origin_reference"]["generated_origin_id"] for row in stage_two_bindings}) == 2
    assert len({tuple(sorted(row["outcome_reference"].items())) for row in stage_two_bindings}) == 2
    assert all(occurrences[row["generation_occurrence_id"]]["occurrence_stage"] == "stage-2" for row in stage_two_bindings)


def test_ordinary_outputs_are_clean_and_deterministic() -> None:
    first = run_generator()
    second = run_generator()
    assert first == second
    joined = b"".join(first.files.values())
    for forbidden in (b"si3_", b"gb3_", b"ex:u_", b"prov:", b"generation_binding"):
        assert forbidden not in joined


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_static_oracle_isolation() -> None:
    native_imports = _imports(ROOT / "src" / "native_reference.py")
    candidate_imports = _imports(ROOT / "src" / "candidate_projection.py")
    assert not any(name.startswith(("generation_relation_core", "experiments.w3c_prov_projection_v1.src.candidate")) for name in native_imports)
    assert not any(name.startswith("experiments.w3c_prov_projection_v1.src.native") for name in candidate_imports)


def test_provn_and_provo_bytes_are_deterministic() -> None:
    first = _run_dual()
    second = _run_dual()
    assert first[3] == second[3]
    assert first[4] == second[4]

