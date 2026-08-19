from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.native_polynomial_oracle import (
    NativePolynomialOracle,
    native_variable_for_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the independent Native N[X] carrier")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    source_path = Path(__file__).parents[1] / "src" / "native_polynomial_oracle.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = sorted(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    identity_x = native_variable_for_source("oracle-source-x")
    identity_y = native_variable_for_source("oracle-source-y")
    x = NativePolynomialOracle.variable(identity_x)
    y = NativePolynomialOracle.variable(identity_y)
    coefficient = x.add(x)
    exponent = x.multiply(x)
    expansion = x.add(y).multiply(x.add(y))
    checks = {
        "zero_additive_identity": x.add(NativePolynomialOracle.zero()) == x,
        "one_multiplicative_identity": x.multiply(NativePolynomialOracle.one()) == x,
        "zero_multiplicative_annihilation": x.multiply(NativePolynomialOracle.zero()) == NativePolynomialOracle.zero(),
        "coefficient_aggregation": coefficient.to_document()["terms"][0]["coefficient"] == 2,
        "exponent_aggregation": exponent.to_document()["terms"][0]["monomial"][0]["exponent"] == 2,
        "binomial_term_count": len(expansion.to_document()["terms"]) == 3,
        "binomial_middle_coefficient": any(
            term["coefficient"] == 2 and len(term["monomial"]) == 2
            for term in expansion.to_document()["terms"]
        ),
    }
    banned = {
        "experiments.provenance_semiring_projection_v1.src.nx_polynomial",
        "experiments.provenance_semiring_projection_v1.src.candidate_nx",
        "experiments.provenance_semiring_projection_v1.src.structural",
        ".nx_polynomial",
        ".candidate_nx",
        ".structural",
    }
    artifact = {
        "schema_version": "independent-native-polynomial-oracle-v1",
        "status": "INDEPENDENT_NATIVE_POLYNOMIAL_PRIMITIVES_SUPPORTED"
        if all(checks.values()) and not (set(imports) & banned)
        else "NOT_ESTABLISHED",
        "implementation": {
            "path": "src/native_polynomial_oracle.py",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "monomial_representation": "frozenset[(variable, exponent)]",
            "polynomial_representation": "immutable tuple of (frozenset monomial, natural coefficient)",
            "imports": imports,
            "banned_import_hits": sorted(set(imports) & banned),
        },
        "checks": checks,
        "vectors": {
            "zero": NativePolynomialOracle.zero().to_document(),
            "one": NativePolynomialOracle.one().to_document(),
            "x_plus_x": coefficient.to_document(),
            "x_times_x": exponent.to_document(),
            "x_plus_y_squared": expansion.to_document(),
        },
        "variable_identity_examples": [
            {"source_identity": "oracle-source-x", "variable": identity_x},
            {"source_identity": "oracle-source-y", "variable": identity_y},
        ],
        "reads_candidate_artifact": False,
        "reads_expected_polynomial": False,
        "reads_core": False,
        "reads_comparison_answer": False,
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "independent_native_polynomial_oracle.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if artifact["status"] == "INDEPENDENT_NATIVE_POLYNOMIAL_PRIMITIVES_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
