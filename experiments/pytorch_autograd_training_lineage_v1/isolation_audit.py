from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from typing import Any

from . import candidate_projection, independent_reference, native_graph


def _module_audit(module: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(module) or "")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    attributes = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Attribute):
            attributes.append(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return {
        "attributes": sorted(set(attributes)),
        "calls": sorted(set(calls)),
        "imports": sorted(set(imports)),
        "module": module.__name__,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_isolation_audit() -> dict[str, Any]:
    candidate = _module_audit(candidate_projection)
    native = _module_audit(native_graph)
    reference = _module_audit(independent_reference)
    candidate_forbidden_attributes = sorted(
        set(candidate["attributes"]) & {
            "grad_fn",
            "next_functions",
            "execution_receipts",
            "native_observation",
        }
    )
    candidate_forbidden_calls = sorted(set(candidate["calls"]) & {"id"})
    candidate_forbidden_imports = sorted(
        item for item in candidate["imports"]
        if item.endswith(("native_graph", "independent_reference")) or item == "torch"
    )
    native_core_imports = sorted(
        item for item in native["imports"] if item.startswith("generation_relation_core")
    )
    reference_core_imports = sorted(
        item for item in reference["imports"] if item.startswith("generation_relation_core")
    )
    return {
        "candidate": candidate,
        "candidate_forbidden_attributes": candidate_forbidden_attributes,
        "candidate_forbidden_calls": candidate_forbidden_calls,
        "candidate_forbidden_imports": candidate_forbidden_imports,
        "candidate_signature": str(inspect.signature(candidate_projection.project_core_to_autograd_graph)),
        "native": native,
        "native_core_imports": native_core_imports,
        "reference": reference,
        "reference_core_imports": reference_core_imports,
        "status": (
            "ORACLE_ISOLATION_VERIFIED"
            if not (
                candidate_forbidden_attributes
                or candidate_forbidden_calls
                or candidate_forbidden_imports
                or native_core_imports
                or reference_core_imports
            )
            else "ORACLE_ISOLATION_FAILED"
        ),
    }
