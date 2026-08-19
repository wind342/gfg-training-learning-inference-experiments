"""Independent pure-state oracle for the frozen native operation sequence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


POSITIVE_KINDS = frozenset(
    {
        "add_multiset",
        "increase_scalar",
        "increment_counter",
        "insert_record",
    }
)
NEGATIVE_KINDS = frozenset(
    {
        "compensate_scalar",
        "decrement_counter",
        "delete_record",
        "remove_multiset",
    }
)
NEUTRAL_KINDS = frozenset({"filter_evaluate"})


def _polynomial_document(
    coefficients: Mapping[str, int], *, integer: bool
) -> dict[str, object]:
    return {
        "schema_version": (
            "zx-polynomial-v1" if integer else "nx-polynomial-v1"
        ),
        "terms": [
            {
                "coefficient": coefficient,
                "monomial": [{"exponent": 1, "variable": variable}],
            }
            for variable, coefficient in sorted(coefficients.items())
            if coefficient
        ],
    }


def _sign_for_kind(kind: str) -> str:
    if kind in POSITIVE_KINDS:
        return "positive"
    if kind in NEGATIVE_KINDS:
        return "negative"
    if kind in NEUTRAL_KINDS:
        return "neutral_or_not_applicable"
    raise ValueError(f"unrecognized frozen operation kind: {kind}")


def _apply_operation(
    engine: str, state: object, operation: dict[str, Any]
) -> object:
    result = deepcopy(state)
    if engine == "sqlite_records":
        records = list(result["records"])
        if operation["kind"] == "insert_record":
            if operation["record_id"] in records:
                raise ValueError("reference insert duplicates a record")
            records.append(operation["record_id"])
        elif operation["kind"] == "delete_record":
            if operation["record_id"] not in records:
                raise ValueError("reference delete misses a record")
            records.remove(operation["record_id"])
        else:
            raise ValueError("unexpected record operation")
        return {"records": sorted(records)}
    if engine == "sqlite_scalar":
        return {"value": int(result["value"]) + int(operation["delta"])}
    if engine == "counter":
        return {
            "counter": int(result["counter"]) + int(operation["delta"])
        }
    if engine == "multiset":
        values = {str(key): int(value) for key, value in result.items()}
        item = operation["item"]
        values[item] = values.get(item, 0) + int(operation["delta"])
        return dict(sorted(values.items()))
    if engine == "filter":
        selected = list(result["selected"])
        keep = operation["item"].startswith(
            operation["required_prefix"]
        )
        if keep:
            selected.append(operation["item"])
        expected_kind = "support" if keep else "disposition"
        if operation["outcome_kind"] != expected_kind:
            raise ValueError("reference filter outcome mismatch")
        return {"selected": selected}
    raise ValueError(f"unsupported reference engine: {engine}")


def evaluate_execution(execution: dict[str, Any]) -> dict[str, Any]:
    engine = execution["engine"]
    if engine == "filter":
        state: object = {"selected": []}
    else:
        state = deepcopy(execution["initial_state"])
    positive: dict[str, int] = {}
    negative: dict[str, int] = {}
    contributions: list[dict[str, Any]] = []
    for operation in execution["operations"]:
        state = _apply_operation(engine, state, operation)
        sign = _sign_for_kind(operation["kind"])
        variable = operation["effect_identity"]
        multiplicity = operation["effect_multiplicity"]
        if sign == "positive":
            positive[variable] = (
                positive.get(variable, 0) + multiplicity
            )
        elif sign == "negative":
            negative[variable] = (
                negative.get(variable, 0) + multiplicity
            )
        contributions.append(
            {
                "effect_identity": variable,
                "multiplicity": multiplicity,
                "occurrence_identity": operation["occurrence_key"],
                "relation_role": operation["relation_role"],
                "sign": sign,
            }
        )
    net = {
        variable: positive.get(variable, 0)
        - negative.get(variable, 0)
        for variable in sorted(set(positive) | set(negative))
    }
    return {
        "case_id": execution["case_id"],
        "contributions": sorted(
            contributions, key=lambda row: row["occurrence_identity"]
        ),
        "execution_id": execution["execution_id"],
        "final_output": state,
        "negative": _polynomial_document(
            negative, integer=False
        ),
        "net": _polynomial_document(net, integer=True),
        "positive": _polynomial_document(
            positive, integer=False
        ),
    }


def evaluate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "executions": [
            evaluate_execution(execution)
            for execution in contract["executions"]
        ],
        "schema_version": "signed-generation-independent-reference-v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    print(
        json.dumps(
            evaluate_contract(contract),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
