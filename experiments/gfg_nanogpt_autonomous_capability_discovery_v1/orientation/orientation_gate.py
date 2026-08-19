from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from ..common import file_sha256, payload_sha256


ORIENTATION_PROMPT = """Read GFG_MACHINE_SEMANTICS.md,
EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md, unrelated_example.json and
ORIENTATION_RECEIPT_CHECKER.py
completely. This executable mechanism-discovery task family is new; use the
guide as the task-method definition rather than silently substituting an
ordinary forecasting or report-writing workflow. The target scientific
nanoGPT GFG is not readable and must not be accessed during this gate.

Write orientation_receipt.json in the repository root with:
- schema = "gfg-orientation-receipt-v1";
- read_complete = true;
- target_gfg_accessed = false;
- a concise distinction between generation_fact and
  generation_occurrence;
- the five components of an atomic generation fact;
- definitions of realizes_fact, reads_from, GeneratedOrigin and
  program_order;
- why equal values do not imply equal facts;
- why facts from one occurrence cannot be Cartesian-recombined;
- why program_order is not data dependence;
- what missing concurrency and missing relations do and do not establish;
- one correct forward training query and one correct reverse training query
  expressed only using the supplied semantics;
- why the executable model rather than the report is the scientific
  authority;
- how to test state sufficiency and closure;
- how multiple prefix cuts prevent retrospective fitting;
- how report claims must map to executable state and update code;
- why merely reading or storing a state field does not make it operational,
  and how executable sensitivity or ablation can test its actual use;
- why intervention analysis must audit every state that can continue to
  evolve; and
- why a repeated pattern in one run is not automatically invariant;
- why formation and stability must be separate executable submodels within
  one composed theory; and
- why a stability model must recursively generate the complete horizon rather
  than only label stability or predict the nearest event.

Use these JSON field names: generation_fact, generation_occurrence,
atomic_generation_fact, realizes_fact, reads_from, GeneratedOrigin,
program_order, equal_values, cartesian_recombination, missing_relations,
forward_query, reverse_query, executable_claim, state_sufficiency,
prefix_only_falsification, report_code_correspondence,
operational_state_use, intervention_state_audit and cross_run_invariance.
Also use dual_dynamics_decomposition and full_horizon_stability.

After writing the receipt, run
python ORIENTATION_RECEIPT_CHECKER.py orientation_receipt.json and revise the
receipt until it prints ORIENTATION_RECEIPT_PASS. This checker validates only
the interface receipt and contains no target evidence or scientific answer.

Do not perform scientific analysis and do not invent a training mechanism.
This is an interface-literacy gate only.
"""


def prepare_orientation_repository(
    *,
    repository: Path,
    orientation_directory: Path,
) -> dict[str, Any]:
    if repository.exists():
        raise RuntimeError("ORIENTATION_REPOSITORY_ALREADY_EXISTS")
    repository.mkdir(parents=True)
    (repository / "submission").mkdir()
    shutil.copy2(
        orientation_directory / "GFG_MACHINE_SEMANTICS.md",
        repository / "GFG_MACHINE_SEMANTICS.md",
    )
    shutil.copy2(
        orientation_directory / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md",
        repository / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md",
    )
    shutil.copy2(
        orientation_directory / "unrelated_example.json",
        repository / "unrelated_example.json",
    )
    shutil.copy2(
        orientation_directory / "ORIENTATION_RECEIPT_CHECKER.py",
        repository / "ORIENTATION_RECEIPT_CHECKER.py",
    )
    (repository / "TASK.txt").write_text(
        ORIENTATION_PROMPT, encoding="utf-8", newline="\n"
    )
    return {
        "orientation_document_sha256": file_sha256(
            repository / "GFG_MACHINE_SEMANTICS.md"
        ),
        "mechanism_discovery_guide_sha256": file_sha256(
            repository / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
        ),
        "orientation_example_sha256": file_sha256(
            repository / "unrelated_example.json"
        ),
        "orientation_receipt_checker_sha256": file_sha256(
            repository / "ORIENTATION_RECEIPT_CHECKER.py"
        ),
        "orientation_prompt_sha256": file_sha256(
            repository / "TASK.txt"
        ),
    }


def validate_orientation_receipt(
    *,
    receipt_path: Path,
    elapsed_seconds: float,
    target_gfg_readable_before_release: bool,
) -> dict[str, Any]:
    if not receipt_path.is_file():
        return {
            "status": "ORIENTATION_FAILURE",
            "reason": "ORIENTATION_RECEIPT_MISSING",
        }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_text = [
        "generation_fact",
        "generation_occurrence",
        "atomic_generation_fact",
        "realizes_fact",
        "reads_from",
        "GeneratedOrigin",
        "program_order",
        "equal_values",
        "cartesian_recombination",
        "missing_relations",
        "forward_query",
        "reverse_query",
        "executable_claim",
        "state_sufficiency",
        "prefix_only_falsification",
        "report_code_correspondence",
        "operational_state_use",
        "intervention_state_audit",
        "cross_run_invariance",
        "dual_dynamics_decomposition",
        "full_horizon_stability",
    ]
    gates = {
        "elapsed_at_least_300_seconds": elapsed_seconds >= 300,
        "receipt_schema": (
            receipt.get("schema") == "gfg-orientation-receipt-v1"
        ),
        "read_complete": receipt.get("read_complete") is True,
        "target_gfg_unreadable_before_release": (
            not target_gfg_readable_before_release
        ),
        "target_gfg_not_accessed": (
            receipt.get("target_gfg_accessed") is False
        ),
        "semantic_fields_complete": all(
            isinstance(receipt.get(name), (str, list, dict))
            and bool(receipt.get(name))
            for name in required_text
        ),
    }
    material = {
        "elapsed_seconds": elapsed_seconds,
        "gates": gates,
        "receipt_sha256": file_sha256(receipt_path),
        "schema": "gfg-orientation-gate-validation-v1",
        "status": (
            "PASS" if all(gates.values()) else "ORIENTATION_FAILURE"
        ),
    }
    material["validation_sha256"] = payload_sha256(material)
    return material
