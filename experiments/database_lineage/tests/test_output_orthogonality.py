import ast
from pathlib import Path

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes
from experiments.database_lineage.src.synthetic_cases import execute_business_query


FORBIDDEN = {
    "tuple_id",
    "origin_id",
    "occurrence_id",
    "binding_id",
    "provenance",
    "lineage",
    "token",
    "stable_tuple_label",
}


def test_contract_on_off_outputs_are_byte_identical_and_clean() -> None:
    disabled, _ = execute_business_query(None)
    adapter = CoreAdapter(run_id="orthogonality")
    enabled, _ = execute_business_query(adapter)
    assert csv_bytes(disabled) == csv_bytes(enabled)
    assert json_bytes(disabled) == json_bytes(enabled)
    assert not (set(enabled[0].values) & FORBIDDEN)
    adapter.validated_snapshot()


def test_tested_path_has_no_evaluator_or_oracle_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    prohibited = {
        "synthetic_oracle",
        "duckdb_reference",
        "provsql_reference",
        "canonical_lineage",
    }
    for name in (
        "relational_executor.py",
        "operators.py",
        "core_adapter.py",
        "core_lineage_reader.py",
    ):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imports = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imports.isdisjoint(prohibited)


def test_oracle_trap_is_not_called_by_tested_execution(monkeypatch) -> None:
    import sys
    import types

    class Trap(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"tested path attempted Oracle access: {name}")

    monkeypatch.setitem(
        sys.modules,
        "experiments.database_lineage.src.synthetic_oracle",
        Trap("experiments.database_lineage.src.synthetic_oracle"),
    )
    adapter = CoreAdapter(run_id="oracle-runtime-trap")
    rows, _executor = execute_business_query(adapter)
    assert rows
    snapshot = adapter.validated_snapshot()
    assert (
        CoreLineageReader(snapshot, adapter.registry)
        .backward(rows[0].tuple_id)
        .tuple_ids
    )
