import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def test_scale_candidate_does_not_import_reference():
    imports = _imports(
        ROOT / "adapters" / "scale_candidate_process.py"
    )
    assert not any("reference" in value for value in imports)


def test_signal_reference_does_not_import_graph_compiler():
    imports = _imports(
        ROOT / "references" / "signal_reference_process.py"
    )
    assert not any("graph_compiler" in value for value in imports)
