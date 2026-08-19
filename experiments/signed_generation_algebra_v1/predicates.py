"""Registered support predicate for the signed-effect query contract."""

from __future__ import annotations


def all_effect_supports(
    support: dict, query: dict, predicate: str
) -> bool:
    return (
        predicate == "membership"
        and query == {}
        and support["outcome_kind"] == "realized_effect"
    )
