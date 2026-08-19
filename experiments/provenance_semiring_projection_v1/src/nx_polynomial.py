from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .structural import variable_for_source


Monomial = tuple[tuple[str, int], ...]


def _canonical_monomial(exponents: Mapping[str, int]) -> Monomial:
    result: list[tuple[str, int]] = []
    for variable, exponent in sorted(exponents.items()):
        if not isinstance(variable, str) or not variable.startswith("x_"):
            raise ValueError(f"invalid N[X] variable: {variable!r}")
        if not isinstance(exponent, int) or isinstance(exponent, bool) or exponent <= 0:
            raise ValueError(f"exponents must be positive integers: {exponent!r}")
        result.append((variable, exponent))
    return tuple(result)


@dataclass(frozen=True)
class NXPolynomial:
    """Canonical finite polynomial over N with explicit coefficients/exponents."""

    coefficients: tuple[tuple[Monomial, int], ...]

    @classmethod
    def zero(cls) -> "NXPolynomial":
        return cls(())

    @classmethod
    def one(cls) -> "NXPolynomial":
        return cls((((), 1),))

    @classmethod
    def variable(cls, variable: str) -> "NXPolynomial":
        return cls(((((variable, 1),), 1),))

    @classmethod
    def from_mapping(cls, values: Mapping[Monomial, int]) -> "NXPolynomial":
        combined: dict[Monomial, int] = {}
        for monomial, coefficient in values.items():
            if not isinstance(coefficient, int) or isinstance(coefficient, bool) or coefficient < 0:
                raise ValueError("N[X] coefficients must be natural numbers")
            normalized = _canonical_monomial(dict(monomial))
            if coefficient:
                combined[normalized] = combined.get(normalized, 0) + coefficient
        return cls(tuple(sorted(combined.items())))

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "NXPolynomial":
        if document.get("schema_version") != "nx-polynomial-v1":
            raise ValueError("unexpected N[X] polynomial schema")
        values: dict[Monomial, int] = {}
        terms = document.get("terms")
        if not isinstance(terms, list):
            raise ValueError("polynomial terms must be a list")
        for term in terms:
            if not isinstance(term, dict):
                raise ValueError("polynomial term must be an object")
            coefficient = term.get("coefficient")
            factors = term.get("monomial")
            if not isinstance(coefficient, int) or isinstance(coefficient, bool) or coefficient <= 0:
                raise ValueError("serialized coefficients must be positive integers")
            if not isinstance(factors, list):
                raise ValueError("serialized monomial must be a list")
            exponent_map: dict[str, int] = {}
            for factor in factors:
                if not isinstance(factor, dict):
                    raise ValueError("serialized factor must be an object")
                variable = factor.get("variable")
                exponent = factor.get("exponent")
                if not isinstance(variable, str) or not isinstance(exponent, int) or isinstance(exponent, bool):
                    raise ValueError("factor requires string variable and integer exponent")
                if variable in exponent_map:
                    raise ValueError("a canonical monomial cannot repeat a variable")
                exponent_map[variable] = exponent
            monomial = _canonical_monomial(exponent_map)
            if monomial in values:
                raise ValueError("a canonical polynomial cannot repeat a monomial")
            values[monomial] = coefficient
        polynomial = cls.from_mapping(values)
        if polynomial.to_document() != document:
            raise ValueError("polynomial document is not canonical")
        return polynomial

    def plus(self, other: "NXPolynomial") -> "NXPolynomial":
        values = dict(self.coefficients)
        for monomial, coefficient in other.coefficients:
            values[monomial] = values.get(monomial, 0) + coefficient
        return self.from_mapping(values)

    def times(self, other: "NXPolynomial") -> "NXPolynomial":
        if not self.coefficients or not other.coefficients:
            return self.zero()
        values: dict[Monomial, int] = {}
        for left_monomial, left_coefficient in self.coefficients:
            for right_monomial, right_coefficient in other.coefficients:
                exponents = dict(left_monomial)
                for variable, exponent in right_monomial:
                    exponents[variable] = exponents.get(variable, 0) + exponent
                monomial = _canonical_monomial(exponents)
                values[monomial] = values.get(monomial, 0) + left_coefficient * right_coefficient
        return self.from_mapping(values)

    @classmethod
    def sum(cls, values: Iterable["NXPolynomial"]) -> "NXPolynomial":
        result = cls.zero()
        for value in values:
            result = result.plus(value)
        return result

    @classmethod
    def product(cls, values: Iterable["NXPolynomial"]) -> "NXPolynomial":
        result = cls.one()
        for value in values:
            result = result.times(value)
        return result

    def variables(self) -> list[str]:
        return sorted({variable for monomial, _ in self.coefficients for variable, _ in monomial})

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "nx-polynomial-v1",
            "terms": [
                {
                    "coefficient": coefficient,
                    "monomial": [
                        {"variable": variable, "exponent": exponent}
                        for variable, exponent in monomial
                    ],
                }
                for monomial, coefficient in self.coefficients
            ],
        }
