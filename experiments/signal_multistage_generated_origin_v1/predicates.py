"""Registered predicates used by the multi-stage signal experiment."""

from __future__ import annotations


def native_key_membership(support: dict, query: dict, predicate: str) -> bool:
    return (
        predicate == "membership"
        and support["native_support_key"] == query["native_support_key"]
    )


def positive_area_rectangle_intersection(
    support: dict, query: dict, predicate: str
) -> bool:
    if predicate != "intersection":
        return False
    left = support["rectangle"]
    right = query["rectangle"]
    return (
        left["x"] < right["x"] + right["width"]
        and right["x"] < left["x"] + left["width"]
        and left["y"] < right["y"] + right["height"]
        and right["y"] < left["y"] + left["height"]
    )

