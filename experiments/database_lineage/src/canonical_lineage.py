from __future__ import annotations

from typing import Iterable


def canonical_tuple_set(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def compare_lineage(
    actual: Iterable[str], expected: Iterable[str]
) -> dict[str, object]:
    actual_set = set(actual)
    expected_set = set(expected)
    return {
        "exact": actual_set == expected_set,
        "actual": sorted(actual_set),
        "expected": sorted(expected_set),
        "false_positives": sorted(actual_set - expected_set),
        "false_negatives": sorted(expected_set - actual_set),
    }
