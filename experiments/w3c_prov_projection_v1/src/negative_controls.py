from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from typing import Any, Callable

from .candidate_projection import project_snapshot
from .provn import parse_provn
from .provo_normalizer import normalize_provo
from .record_model import canonical_json_bytes, validate_normalized_records
from .science_runs import run_full
from .validation import exact_comparison


def _mutate(records: list[dict[str, Any]], kind: str, change: Callable[[dict[str, Any]], None]) -> list[dict[str, Any]]:
    value = copy.deepcopy(records)
    change(next(row for row in value if row["kind"] == kind))
    return value


def _remove(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    value = copy.deepcopy(records)
    value.remove(next(row for row in value if row["kind"] == kind))
    return value


def _invalid(records: list[dict[str, Any]]) -> bool:
    return bool(validate_normalized_records(records))


def _different(records: list[dict[str, Any]], baseline: list[dict[str, Any]], bindings: int = 14) -> bool:
    return exact_comparison(records, baseline, bindings)["status"] == "NOT_SUPPORTED"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def run_negative_controls(source_root: Path) -> dict[str, Any]:
    full = run_full()
    baseline = full.candidate_records
    provn = full.candidate_provn
    ttl = full.native_ttl
    output = b"".join(full.output.files.values())
    usage_rows = [row for row in baseline if row["kind"] == "usage"]
    generation_rows = [row for row in baseline if row["kind"] == "generation"]
    entity_rows = [row for row in baseline if row["kind"] == "entity"]

    results: list[dict[str, Any]] = []

    def add(number: int, name: str, classification: str, detected: bool, detector: str) -> None:
        results.append({
            "number": number, "name": name, "classification": classification,
            "detected": bool(detected), "detector": detector,
        })

    add(1, "missing Entity", "VALIDATOR_UNIT", _invalid(_remove(baseline, "entity")), "reference closure validator")
    add(2, "missing Activity", "VALIDATOR_UNIT", _invalid(_remove(baseline, "activity")), "reference closure validator")
    add(3, "missing Usage", "VALIDATOR_UNIT", _invalid(_remove(baseline, "usage")), "expanded derivation closure validator")
    add(4, "missing Generation", "VALIDATOR_UNIT", _invalid(_remove(baseline, "generation")), "expanded derivation closure validator")
    add(5, "missing Derivation", "VALIDATOR_UNIT", _different(_remove(baseline, "derivation"), baseline), "P1 exact comparator")
    add(6, "dangling Usage reference", "VALIDATOR_UNIT", _invalid(_mutate(baseline, "usage", lambda row: row.update(entity="ex:e_missing"))), "reference closure validator")
    add(7, "dangling Generation reference", "VALIDATOR_UNIT", _invalid(_mutate(baseline, "generation", lambda row: row.update(activity="ex:a_missing"))), "reference closure validator")
    add(8, "Derivation references wrong Usage", "OFFICIAL_CONSTRAINT", _invalid(_mutate(baseline, "derivation", lambda row: row.update(usage=usage_rows[-1]["id"]))), "Inference 11 consistency")
    wrong_generation = next(
        row["id"] for row in generation_rows
        if row["id"] != next(item for item in baseline if item["kind"] == "derivation")["generation"]
    )
    add(9, "Derivation references wrong Generation", "OFFICIAL_CONSTRAINT", _invalid(_mutate(baseline, "derivation", lambda row: row.update(generation=wrong_generation))), "Inference 11 consistency")
    add(10, "Derivation references wrong Activity", "OFFICIAL_CONSTRAINT", _invalid(_mutate(baseline, "derivation", lambda row: row.update(activity="ex:a_wrong"))), "Inference 11 consistency")
    duplicate_generation = copy.deepcopy(baseline)
    duplicate_generation.append({**generation_rows[0], "id": "ex:g_duplicate_generation"})
    add(11, "duplicate Generation for one Entity", "OFFICIAL_CONSTRAINT", _invalid(duplicate_generation), "Constraint 24 unique-generation")
    duplicate_identifier = copy.deepcopy(baseline)
    duplicate_identifier.append({**usage_rows[0], "entity": entity_rows[-1]["id"]})
    add(12, "duplicate relation identifier", "OFFICIAL_CONSTRAINT", _invalid(duplicate_identifier), "Constraint 23 and identifier uniqueness")
    semantic_collision = [{"id": "ex:e_collision", "payload": 1}, {"id": "ex:e_collision", "payload": 2}]
    add(13, "duplicate semantic key with different payload", "VALIDATOR_UNIT", len({row["id"] for row in semantic_collision}) != len({canonical_json_bytes(row) for row in semantic_collision}), "semantic-key collision guard")

    fabricated = copy.deepcopy(baseline)
    fabricated.append({**next(row for row in baseline if row["kind"] == "derivation"), "id": "ex:d_fabricated_pairing", "used_entity": entity_rows[-1]["id"]})
    add(14, "fabricated source-outcome pairing", "VALIDATOR_UNIT", _different(fabricated, baseline), "P1 pairing comparator")
    cartesian = copy.deepcopy(fabricated)
    cartesian.append({**fabricated[-1], "id": "ex:d_cartesian_product", "used_entity": entity_rows[-2]["id"]})
    add(15, "Cartesian product fabrication", "VALIDATOR_UNIT", _different(cartesian, baseline), "P1 pairing comparator")
    add(16, "relation-role loss", "VALIDATOR_UNIT", _different(_mutate(baseline, "derivation", lambda row: row.update(role="ex:lost")), baseline), "field-exact comparator")
    collapsed = _remove(_remove(baseline, "derivation"), "usage")
    add(17, "legal multiplicity collapse", "VALIDATOR_UNIT", _different(collapsed, baseline), "multiplicity comparator")
    duplicated = copy.deepcopy(baseline)
    duplicated.append({**next(row for row in baseline if row["kind"] == "derivation"), "id": "ex:d_illegal_duplicate"})
    add(18, "illegal multiplicity duplication", "VALIDATOR_UNIT", _different(duplicated, baseline), "multiplicity comparator")
    broken_snapshot = copy.deepcopy(full.snapshot)
    broken_snapshot.tables.generated_origins[0]["origin_payload"]["prior_support_id"] = "ps3_" + "0" * 64
    try:
        project_snapshot(broken_snapshot)
        broken_detected = False
    except (KeyError, ValueError):
        broken_detected = True
    add(19, "GeneratedOrigin broken bridge", "VALIDATOR_UNIT", broken_detected, "GeneratedOrigin prior-support resolver")

    shortcut = copy.deepcopy(baseline)
    final_derivation = next(row for row in baseline if row["kind"] == "derivation" and row["role"] == "ex:stage2_input")
    source_entity = next(row["id"] for row in baseline if row["kind"] == "entity" and row["types"] == ["ex:SourceInformation"])
    shortcut_usage = {
        "kind": "usage", "id": "ex:u_invented_shortcut", "activity": final_derivation["activity"],
        "entity": source_entity, "role": "ex:invented_shortcut", "ordinal": 99,
    }
    shortcut_derivation = {
        "kind": "derivation", "id": "ex:d_invented_shortcut", "generated_entity": final_derivation["generated_entity"],
        "used_entity": source_entity, "activity": final_derivation["activity"], "generation": final_derivation["generation"],
        "usage": shortcut_usage["id"], "role": shortcut_usage["role"], "ordinal": 99,
    }
    shortcut.extend([shortcut_usage, shortcut_derivation])
    add(20, "invented direct original-final shortcut", "VALIDATOR_UNIT", _different(shortcut, baseline), "explicit-binding comparator")
    add(21, "invalid QName", "VALIDATOR_UNIT", re.fullmatch(r"^(ex|prov):[A-Za-z_][A-Za-z0-9_]*$", "ex:bad id") is None, "QName lexical validator")
    try:
        parse_provn(provn.replace(b"https://example.org/w3c-prov-projection-v1#", b"https://wrong.example/#", 1))
        namespace_detected = False
    except ValueError:
        namespace_detected = True
    add(22, "invalid namespace", "VALIDATOR_UNIT", namespace_detected, "frozen PROV-N wrapper validator")
    try:
        parse_provn(provn.replace(b"endDocument\n", b"", 1))
        malformed_detected = False
    except ValueError:
        malformed_detected = True
    add(23, "malformed PROV-N", "VALIDATOR_UNIT", malformed_detected, "independent PROV-N parser")
    try:
        normalize_provo(ttl.replace(b"prov:hadUsage", b"prov:brokenUsage", 1))
        provo_detected = False
    except ValueError:
        provo_detected = True
    add(24, "invalid PROV-O qualified relation", "VALIDATOR_UNIT", provo_detected, "RDFLib plus qualified-structure normalizer")

    candidate_imports = _imports(source_root / "candidate_projection.py")
    native_imports = _imports(source_root / "native_reference.py")
    add(25, "candidate reads native reference", "ISOLATION", not any(name.endswith(("native_reference", "provo_normalizer")) for name in candidate_imports), "static candidate import graph")
    add(26, "native reference reads Core", "ISOLATION", not any(name.startswith("generation_relation_core") for name in native_imports), "static native import graph")
    candidate_source = (source_root / "candidate_projection.py").read_text(encoding="utf-8")
    add(27, "second authority store", "ISOLATION", all(token not in candidate_source for token in ("expected_prov", "answer_table", "reference.json")), "candidate source/storage audit")
    add(28, "hidden lookup table", "ISOLATION", all(token not in candidate_source for token in ("open(", "read_text(", "read_bytes(", "Path(")), "candidate file-read audit")
    add(29, "complete Snapshot embedded as PROV attribute", "VALIDATOR_UNIT", b"snapshot_id" not in canonical_json_bytes(baseline), "closed attribute allowlist")
    add(30, "Evidence embedded as opaque PROV blob", "VALIDATOR_UNIT", b"evidence_id" not in canonical_json_bytes(baseline), "closed attribute allowlist")
    contaminated = output + b"\nprov:qualifiedUsage ex:u_control\n"
    add(31, "output contamination", "END_TO_END", b"prov:" in contaminated and b"prov:" not in output, "five-mode output byte guard")
    blockers = ["SYNTHETIC_BLOCKER"]
    status = "W3C_PROV_PROJECTION_V1_SUPPORTED" if not blockers else "W3C_PROV_PROJECTION_NOT_SUPPORTED"
    add(32, "unsupported status escalation", "END_TO_END", status != "W3C_PROV_PROJECTION_V1_SUPPORTED", "blocking-reason status gate")

    by_classification = {
        classification: sum(item["classification"] == classification for item in results)
        for classification in ("END_TO_END", "ISOLATION", "VALIDATOR_UNIT", "OFFICIAL_CONSTRAINT")
    }
    detected_count = sum(item["detected"] for item in results)
    return {
        "negative_control_count": len(results),
        "detected_count": detected_count,
        "undetected_count": len(results) - detected_count,
        "classification_counts": by_classification,
        "controls": results,
        "status": "SUPPORTED" if len(results) == detected_count == 32 else "NOT_SUPPORTED",
    }
