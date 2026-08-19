from __future__ import annotations

import ast
import hashlib
import inspect

from experiments.provenance_semiring_projection_v1.src import native_polynomial_oracle
from experiments.provenance_semiring_projection_v1.src import candidate_nx, native_nx
from experiments.provenance_semiring_projection_v1.src.native_polynomial_oracle import (
    NativePolynomialOracle,
    native_variable_for_source,
)


X = "x_" + "1" * 64
Y = "x_" + "2" * 64


def test_independent_native_polynomial_zero_and_one() -> None:
    x = NativePolynomialOracle.variable(X)
    assert NativePolynomialOracle.zero().add(x) == x
    assert NativePolynomialOracle.one().multiply(x) == x
    assert NativePolynomialOracle.zero().multiply(x) == NativePolynomialOracle.zero()


def test_independent_native_polynomial_addition_merges_coefficients() -> None:
    x = NativePolynomialOracle.variable(X)
    assert x.add(x).to_document() == {
        "schema_version": "nx-polynomial-v1",
        "terms": [{"coefficient": 2, "monomial": [{"variable": X, "exponent": 1}]}],
    }


def test_independent_native_polynomial_multiplication_merges_exponents() -> None:
    x = NativePolynomialOracle.variable(X)
    y = NativePolynomialOracle.variable(Y)
    product = x.multiply(x).multiply(y)
    assert product.to_document() == {
        "schema_version": "nx-polynomial-v1",
        "terms": [
            {
                "coefficient": 1,
                "monomial": [
                    {"variable": X, "exponent": 2},
                    {"variable": Y, "exponent": 1},
                ],
            }
        ],
    }


def test_independent_native_polynomial_expansion_aggregates_middle_term() -> None:
    x = NativePolynomialOracle.variable(X)
    y = NativePolynomialOracle.variable(Y)
    terms = x.add(y).multiply(x.add(y)).to_document()["terms"]
    assert len(terms) == 3
    assert any(term["coefficient"] == 2 and len(term["monomial"]) == 2 for term in terms)


def test_native_variable_identity_is_locally_implemented() -> None:
    source = "complete-source-identity"
    expected = "x_" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert native_variable_for_source(source) == expected


def test_native_oracle_has_no_candidate_algebra_or_structural_import() -> None:
    tree = ast.parse(inspect.getsource(native_polynomial_oracle))
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name.endswith(("nx_polynomial", "candidate_nx", "structural"))
        for name in imported
    )


def test_native_and_candidate_use_different_algebra_modules() -> None:
    native_source = inspect.getsource(native_nx)
    candidate_source = inspect.getsource(candidate_nx)
    assert "NativePolynomialOracle" in native_source
    assert "NXPolynomial" not in native_source
    assert "NXPolynomial" in candidate_source
    assert "NativePolynomialOracle" not in candidate_source
