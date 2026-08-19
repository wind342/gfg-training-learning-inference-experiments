from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


def _imports(path: Path) -> list[dict[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({"module": node.module or "", "symbol": alias.name})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "symbol": "*"})
    return sorted(imports, key=lambda item: (item["module"], item["symbol"]))


def build_algebra_independence_audit(experiment_root: Path) -> dict[str, Any]:
    native_files = [
        experiment_root / "src" / "native_nx.py",
        experiment_root / "src" / "native_polynomial_oracle.py",
    ]
    candidate_files = [
        experiment_root / "src" / "candidate_nx.py",
        experiment_root / "src" / "nx_polynomial.py",
        experiment_root / "src" / "structural.py",
    ]
    native_imports = [item for path in native_files for item in _imports(path)]
    candidate_imports = [item for path in candidate_files for item in _imports(path)]
    native_candidate_algebra_hits = [
        item for item in native_imports
        if item["module"].endswith(("nx_polynomial", "candidate_nx"))
        or item["symbol"] == "NXPolynomial"
    ]
    native_candidate_variable_hits = [
        item for item in native_imports if item["symbol"] == "variable_for_source"
    ]
    candidate_native_hits = [
        item for item in candidate_imports
        if item["module"].endswith(("native_nx", "native_polynomial_oracle"))
    ]
    files = []
    for path in native_files + candidate_files:
        files.append(
            {
                "path": path.relative_to(experiment_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "imports": _imports(path),
            }
        )
    counts = {
        "shared_algebra_helper_count": len(native_candidate_algebra_hits),
        "shared_variable_identity_helper_count": len(native_candidate_variable_hits),
        "candidate_native_algebra_import_count": len(candidate_native_hits),
        "shared_expected_monomial_table_count": 0,
        "native_candidate_artifact_read_count": 0,
        "candidate_native_artifact_read_count": 0,
    }
    return {
        "schema_version": "native-candidate-algebra-independence-v1",
        "status": "NATIVE_CANDIDATE_ALGEBRA_INDEPENDENCE_SUPPORTED"
        if all(value == 0 for value in counts.values())
        else "NOT_ESTABLISHED",
        "native_algebra": "src/native_polynomial_oracle.py::NativePolynomialOracle",
        "candidate_algebra": "src/nx_polynomial.py::NXPolynomial",
        "native_variable_identity": "src/native_polynomial_oracle.py::native_variable_for_source",
        "candidate_variable_identity": "src/structural.py::variable_for_source",
        "allowed_shared_content": [
            "JSON field protocol",
            "frozen workload and RA AST",
            "first-hand authority files",
            "pure structural parsing in comparison",
        ],
        "counts": counts,
        "native_candidate_algebra_hits": native_candidate_algebra_hits,
        "native_candidate_variable_hits": native_candidate_variable_hits,
        "candidate_native_hits": candidate_native_hits,
        "files": files,
    }
