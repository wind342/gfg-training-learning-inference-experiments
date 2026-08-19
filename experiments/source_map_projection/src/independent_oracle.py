"""Evaluation-only frozen Oracle. Tested generator/collector/projection imports are forbidden."""

from __future__ import annotations

from pathlib import Path


def oracle_utf16_column(prefix: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in prefix)


def oracle_position(text: str, index: int) -> dict[str, int]:
    line = 0
    line_start = 0
    cursor = 0
    while cursor < index:
        if text.startswith("\r\n", cursor):
            cursor += 2
            line += 1
            line_start = cursor
        elif text[cursor] in "\r\n":
            cursor += 1
            line += 1
            line_start = cursor
        else:
            cursor += 1
    return {"line": line, "column": oracle_utf16_column(text[line_start:index])}


def _source_position(text: str, fragment: str, occurrence: int = 0) -> dict[str, int]:
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.find(fragment, cursor)
        if start < 0:
            raise AssertionError(f"Oracle fragment missing: {fragment!r}")
        cursor = start + len(fragment)
    return oracle_position(text, start)


def adversarial_oracle(fixture_dir: Path) -> dict:
    first = (fixture_dir / "adversarial-a.js").read_text(encoding="utf-8")
    second = (fixture_dir / "adversarial-b.js").read_text(encoding="utf-8")
    chunks = [
        ("(function(){\n", None),
        ('const highlight = "你好🔥";\n', ("adversarial-a.js", first, 'const café = "你好🔥";', "café")),
        ("const value = 7;\n", ("adversarial-b.js", second, "const value = 7;", None)),
        ('const label = "总计";\n', ("adversarial-b.js", second, 'const label = "总计";', None)),
        ('const combine = "é";\n', ("adversarial-a.js", first, 'const combine = "é";', None)),
        ("const duplicated = [", None),
        ('"你好🔥"', ("adversarial-a.js", first, '"你好🔥"', None)),
        (",", None),
        ('"你好🔥"', ("adversarial-a.js", first, '"你好🔥"', None)),
        ("];\n", None),
        ('function show(item) {\n\treturn `${label}:${item}`;\n}\n', ("adversarial-b.js", second, 'function show(item) {\n\treturn `${label}:${item}`;\n}', None)),
        ("function add(left,right){return left+right;}\n", ("adversarial-a.js", first, 'function add(left, right) {\n  return left + right;\n}', None)),
        ("const inserted = 99;\n", None),
        ("console.log(highlight, value, combine, duplicated, inserted);\n", ("adversarial-a.js", first, "café", "café")),
        ("})();\n", None),
    ]
    output = ""
    records = []
    for chunk, source in chunks:
        generated = oracle_position(output, len(output))
        output += chunk
        if source is None:
            records.append({
                "generated_file": "adversarial-output.js", **{f"generated_{key}": value for key, value in generated.items()},
                "mapped": False, "original_source": None, "original_line": None,
                "original_column": None, "original_name": None,
            })
        else:
            path, text, fragment, name = source
            original = _source_position(text, fragment)
            records.append({
                "generated_file": "adversarial-output.js", **{f"generated_{key}": value for key, value in generated.items()},
                "mapped": True, "original_source": path,
                "original_line": original["line"], "original_column": original["column"],
                "original_name": name,
            })
    return {"output_bytes": output.encode("utf-8"), "records": records}
