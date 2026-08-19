from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


NativeMonomial = frozenset[tuple[str, int]]


def native_variable_for_source(source_identity: str) -> str:
    """Derive the frozen variable identity without any Candidate helper."""

    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("source identity must be a non-empty string")
    return "x_" + hashlib.sha256(source_identity.encode("utf-8")).hexdigest()


def _normalize_monomial(monomial: NativeMonomial) -> NativeMonomial:
    exponents: dict[str, int] = {}
    for variable, exponent in monomial:
        if not isinstance(variable, str) or not variable.startswith("x_"):
            raise ValueError(f"invalid N[X] variable: {variable!r}")
        if not isinstance(exponent, int) or isinstance(exponent, bool) or exponent <= 0:
            raise ValueError(f"exponents must be positive integers: {exponent!r}")
        if variable in exponents:
            raise ValueError("native monomial repeats a variable")
        exponents[variable] = exponent
    return frozenset(exponents.items())


def _monomial_key(monomial: NativeMonomial) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(monomial))


@dataclass(frozen=True)
class NativePolynomialOracle:
    """Independent finite N[X] implementation using frozenset monomials.

    This module deliberately does not import the Candidate polynomial module,
    Candidate projection, Core, structural helpers, or expected artifacts.
    """

    coefficients: tuple[tuple[NativeMonomial, int], ...]

    @classmethod
    def zero(cls) -> "NativePolynomialOracle":
        return cls(())

    @classmethod
    def one(cls) -> "NativePolynomialOracle":
        return cls(((frozenset(), 1),))

    @classmethod
    def variable(cls, variable: str) -> "NativePolynomialOracle":
        return cls.from_mapping({frozenset({(variable, 1)}): 1})

    @classmethod
    def from_mapping(
        cls, values: Mapping[NativeMonomial, int]
    ) -> "NativePolynomialOracle":
        combined: dict[NativeMonomial, int] = {}
        for raw_monomial, coefficient in values.items():
            if not isinstance(coefficient, int) or isinstance(coefficient, bool) or coefficient < 0:
                raise ValueError("native N[X] coefficients must be natural numbers")
            monomial = _normalize_monomial(raw_monomial)
            if coefficient:
                combined[monomial] = combined.get(monomial, 0) + coefficient
        ordered = tuple(
            sorted(combined.items(), key=lambda item: _monomial_key(item[0]))
        )
        return cls(ordered)

    def add(self, other: "NativePolynomialOracle") -> "NativePolynomialOracle":
        values = dict(self.coefficients)
        for monomial, coefficient in other.coefficients:
            values[monomial] = values.get(monomial, 0) + coefficient
        return self.from_mapping(values)

    def multiply(self, other: "NativePolynomialOracle") -> "NativePolynomialOracle":
        if not self.coefficients or not other.coefficients:
            return self.zero()
        values: dict[NativeMonomial, int] = {}
        for left_monomial, left_coefficient in self.coefficients:
            for right_monomial, right_coefficient in other.coefficients:
                exponents = dict(left_monomial)
                for variable, exponent in right_monomial:
                    exponents[variable] = exponents.get(variable, 0) + exponent
                monomial = frozenset(exponents.items())
                values[monomial] = (
                    values.get(monomial, 0)
                    + left_coefficient * right_coefficient
                )
        return self.from_mapping(values)

    @classmethod
    def add_all(
        cls, values: Iterable["NativePolynomialOracle"]
    ) -> "NativePolynomialOracle":
        result = cls.zero()
        for value in values:
            result = result.add(value)
        return result

    @classmethod
    def multiply_all(
        cls, values: Iterable["NativePolynomialOracle"]
    ) -> "NativePolynomialOracle":
        result = cls.one()
        for value in values:
            result = result.multiply(value)
        return result

    def variable_names(self) -> list[str]:
        return sorted(
            {
                variable
                for monomial, _coefficient in self.coefficients
                for variable, _exponent in monomial
            }
        )

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "nx-polynomial-v1",
            "terms": [
                {
                    "coefficient": coefficient,
                    "monomial": [
                        {"variable": variable, "exponent": exponent}
                        for variable, exponent in _monomial_key(monomial)
                    ],
                }
                for monomial, coefficient in self.coefficients
            ],
        }
