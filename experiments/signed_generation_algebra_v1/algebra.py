"""Canonical unreduced signed pairs over finite N[X] polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


Monomial = tuple[tuple[str, int], ...]


def _canonical_monomial(exponents: Mapping[str, int]) -> Monomial:
    factors: list[tuple[str, int]] = []
    for variable, exponent in sorted(exponents.items()):
        if not isinstance(variable, str) or not variable.startswith("x_"):
            raise ValueError(f"invalid polynomial variable: {variable!r}")
        if (
            not isinstance(exponent, int)
            or isinstance(exponent, bool)
            or exponent <= 0
        ):
            raise ValueError("polynomial exponents must be positive integers")
        factors.append((variable, exponent))
    return tuple(factors)


@dataclass(frozen=True)
class NaturalPolynomial:
    """Canonical finite polynomial over N with no zero-coefficient terms."""

    coefficients: tuple[tuple[Monomial, int], ...]

    @classmethod
    def zero(cls) -> "NaturalPolynomial":
        return cls(())

    @classmethod
    def one(cls) -> "NaturalPolynomial":
        return cls((((), 1),))

    @classmethod
    def variable(
        cls, variable: str, coefficient: int = 1
    ) -> "NaturalPolynomial":
        return cls.from_mapping({((variable, 1),): coefficient})

    @classmethod
    def from_mapping(
        cls, values: Mapping[Monomial, int]
    ) -> "NaturalPolynomial":
        combined: dict[Monomial, int] = {}
        for monomial, coefficient in values.items():
            if (
                not isinstance(coefficient, int)
                or isinstance(coefficient, bool)
                or coefficient < 0
            ):
                raise ValueError("N[X] coefficients must be natural numbers")
            normalized = _canonical_monomial(dict(monomial))
            if coefficient:
                combined[normalized] = (
                    combined.get(normalized, 0) + coefficient
                )
        return cls(tuple(sorted(combined.items())))

    def plus(self, other: "NaturalPolynomial") -> "NaturalPolynomial":
        values = dict(self.coefficients)
        for monomial, coefficient in other.coefficients:
            values[monomial] = values.get(monomial, 0) + coefficient
        return self.from_mapping(values)

    def times(self, other: "NaturalPolynomial") -> "NaturalPolynomial":
        values: dict[Monomial, int] = {}
        for left_monomial, left_coefficient in self.coefficients:
            for right_monomial, right_coefficient in other.coefficients:
                exponents = dict(left_monomial)
                for variable, exponent in right_monomial:
                    exponents[variable] = (
                        exponents.get(variable, 0) + exponent
                    )
                monomial = _canonical_monomial(exponents)
                values[monomial] = (
                    values.get(monomial, 0)
                    + left_coefficient * right_coefficient
                )
        return self.from_mapping(values)

    @classmethod
    def sum(
        cls, values: Iterable["NaturalPolynomial"]
    ) -> "NaturalPolynomial":
        result = cls.zero()
        for value in values:
            result = result.plus(value)
        return result

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


@dataclass(frozen=True)
class IntegerPolynomial:
    """Canonical finite polynomial over Z with zero terms removed."""

    coefficients: tuple[tuple[Monomial, int], ...]

    @classmethod
    def from_mapping(
        cls, values: Mapping[Monomial, int]
    ) -> "IntegerPolynomial":
        normalized: dict[Monomial, int] = {}
        for monomial, coefficient in values.items():
            if not isinstance(coefficient, int) or isinstance(
                coefficient, bool
            ):
                raise ValueError("Z[X] coefficients must be integers")
            canonical = _canonical_monomial(dict(monomial))
            if coefficient:
                normalized[canonical] = (
                    normalized.get(canonical, 0) + coefficient
                )
        return cls(
            tuple(
                (monomial, coefficient)
                for monomial, coefficient in sorted(normalized.items())
                if coefficient
            )
        )

    def plus(self, other: "IntegerPolynomial") -> "IntegerPolynomial":
        values = dict(self.coefficients)
        for monomial, coefficient in other.coefficients:
            values[monomial] = values.get(monomial, 0) + coefficient
        return self.from_mapping(values)

    def times(self, other: "IntegerPolynomial") -> "IntegerPolynomial":
        values: dict[Monomial, int] = {}
        for left_monomial, left_coefficient in self.coefficients:
            for right_monomial, right_coefficient in other.coefficients:
                exponents = dict(left_monomial)
                for variable, exponent in right_monomial:
                    exponents[variable] = (
                        exponents.get(variable, 0) + exponent
                    )
                monomial = _canonical_monomial(exponents)
                values[monomial] = (
                    values.get(monomial, 0)
                    + left_coefficient * right_coefficient
                )
        return self.from_mapping(values)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "zx-polynomial-v1",
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


@dataclass(frozen=True)
class SignedPair:
    """Unreduced positive/negative pair; no cross-component cancellation."""

    positive: NaturalPolynomial
    negative: NaturalPolynomial

    @classmethod
    def zero(cls) -> "SignedPair":
        return cls(NaturalPolynomial.zero(), NaturalPolynomial.zero())

    @classmethod
    def one(cls) -> "SignedPair":
        return cls(NaturalPolynomial.one(), NaturalPolynomial.zero())

    @classmethod
    def positive_variable(
        cls, variable: str, coefficient: int = 1
    ) -> "SignedPair":
        return cls(
            NaturalPolynomial.variable(variable, coefficient),
            NaturalPolynomial.zero(),
        )

    @classmethod
    def negative_variable(
        cls, variable: str, coefficient: int = 1
    ) -> "SignedPair":
        return cls(
            NaturalPolynomial.zero(),
            NaturalPolynomial.variable(variable, coefficient),
        )

    def plus(self, other: "SignedPair") -> "SignedPair":
        return SignedPair(
            self.positive.plus(other.positive),
            self.negative.plus(other.negative),
        )

    def times(self, other: "SignedPair") -> "SignedPair":
        return SignedPair(
            self.positive.times(other.positive).plus(
                self.negative.times(other.negative)
            ),
            self.positive.times(other.negative).plus(
                self.negative.times(other.positive)
            ),
        )

    def negated(self) -> "SignedPair":
        return SignedPair(self.negative, self.positive)

    def net_projection(self) -> IntegerPolynomial:
        coefficients = dict(self.positive.coefficients)
        for monomial, coefficient in self.negative.coefficients:
            coefficients[monomial] = (
                coefficients.get(monomial, 0) - coefficient
            )
        return IntegerPolynomial.from_mapping(coefficients)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "unreduced-signed-pair-v1",
            "positive": self.positive.to_document(),
            "negative": self.negative.to_document(),
        }


def signed_sum(values: Iterable[SignedPair]) -> SignedPair:
    result = SignedPair.zero()
    for value in values:
        result = result.plus(value)
    return result
