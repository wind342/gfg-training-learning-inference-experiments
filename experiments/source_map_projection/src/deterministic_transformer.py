from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .transformation_dsl import Emitter, OriginSpan, SourceDocument, TransformResult, finalize


def _template_origin(text: str) -> OriginSpan:
    document = SourceDocument.from_text("generator-template-v1.js", text)
    return OriginSpan(document, 0, len(text)).configured(
        mapping_anchor=False,
        relation_role="participation:generator_template",
    )


def _emit_synthetic(emitter: Emitter, text: str, operation: str) -> dict:
    return emitter.emit(
        text,
        operation_type=operation,
        origins=[_template_origin(text)],
        mapping_eligible=False,
    )


def materialize_generated_inputs(contract_path: Path, output_dir: Path) -> dict[str, SourceDocument]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    crlf = contract["crlf_fixture"]
    crlf_text = "\r\n".join(crlf["lines"])
    crlf_doc = SourceDocument.from_text(crlf["path"], crlf_text)
    (output_dir / crlf["path"]).write_bytes(crlf_doc.raw_bytes)
    result = {crlf_doc.logical_path: crlf_doc}
    medium = contract["medium_fixture"]
    for source in range(medium["source_file_count"]):
        name = f"medium-{source}.js"
        lines = [
            medium["line_template"].format(source=source, index=index, value=source * 1000 + index)
            for index in range(medium["records_per_file"])
        ]
        text = "\n".join(lines) + ("\n" if medium["trailing_newline"] else "")
        document = SourceDocument.from_text(name, text)
        (output_dir / name).write_bytes(document.raw_bytes)
        result[name] = document
    return result


def adversarial_transform(
    fixture_dir: Path,
    *,
    run_id: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    first = SourceDocument.read(fixture_dir / "adversarial-a.js")
    second = SourceDocument.read(fixture_dir / "adversarial-b.js")
    emitter = Emitter(run_id, "adversarial", "adversarial-output.js", "../src", observer, record_receipts)
    _emit_synthetic(emitter, "(function(){\n", "EMIT_SYNTHETIC_WRAPPER")
    rename = first.unique_span('const café = "你好🔥";')
    emitter.emit(
        'const highlight = "你好🔥";\n',
        operation_type="RENAME_IDENTIFIER",
        origins=[rename.configured(original_name="café", generated_name="highlight", relation_role="source_map_anchor:RENAME_IDENTIFIER")],
        mapping_eligible=True,
        transform_parameters={"from": "café", "to": "highlight"},
    )
    for fragment in ('const value = 7;', 'const label = "总计";'):
        emitter.emit(
            second.unique_span(fragment).text + "\n",
            operation_type="COPY_RANGE",
            origins=[second.unique_span(fragment)],
            mapping_eligible=True,
        )
    combine = first.unique_span('const combine = "é";')
    emitter.emit(combine.text + "\n", operation_type="COPY_RANGE", origins=[combine], mapping_eligible=True)
    _emit_synthetic(emitter, "const duplicated = [", "INSERT_LITERAL")
    string_span = first.unique_span('"你好🔥"')
    emitter.emit(string_span.text, operation_type="DUPLICATE_RANGE", origins=[string_span.configured(relation_role="source_map_anchor:DUPLICATE_RANGE")], mapping_eligible=True)
    _emit_synthetic(emitter, ",", "INSERT_LITERAL")
    emitter.emit(string_span.text, operation_type="DUPLICATE_RANGE", origins=[string_span.configured(relation_role="source_map_anchor:DUPLICATE_RANGE")], mapping_eligible=True)
    _emit_synthetic(emitter, "];\n", "INSERT_LITERAL")
    show = second.unique_span('function show(item) {\n\treturn `${label}:${item}`;\n}')
    emitter.emit(show.text + "\n", operation_type="REORDER_RANGE", origins=[show.configured(relation_role="source_map_anchor:REORDER_RANGE")], mapping_eligible=True)
    add = first.unique_span('function add(left, right) {\n  return left + right;\n}')
    emitter.emit(
        "function add(left,right){return left+right;}\n",
        operation_type="COLLAPSE_WHITESPACE",
        origins=[add.configured(relation_role="source_map_anchor:COLLAPSE_WHITESPACE")],
        mapping_eligible=True,
    )
    _emit_synthetic(emitter, "const inserted = 99;\n", "INSERT_LITERAL")
    primary = first.unique_span("café").configured(
        original_name="café", generated_name="highlight", relation_role="source_map_anchor:CONCATENATE_SOURCES"
    )
    secondary = second.unique_span("value").configured(
        mapping_anchor=False, relation_role="participation:CONCATENATE_SOURCES_secondary"
    )
    emitter.emit(
        "console.log(highlight, value, combine, duplicated, inserted);\n",
        operation_type="CONCATENATE_SOURCES",
        origins=[primary, secondary],
        mapping_eligible=True,
    )
    emitter.dispose(first.unique_span("// DELETE_ME\n"))
    _emit_synthetic(emitter, "})();\n", "EMIT_SYNTHETIC_WRAPPER")
    return finalize(emitter)


def unicode_crlf_transform(
    document: SourceDocument,
    *,
    run_id: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    emitter = Emitter(run_id, "unicode_crlf", "unicode-crlf-output.js", "../src", observer, record_receipts)
    for span in document.line_spans():
        if span.text in {"\r\n", "\n", ""}:
            _emit_synthetic(emitter, span.text or "\n", "INSERT_LITERAL")
        else:
            emitter.emit(span.text, operation_type="COPY_RANGE", origins=[span], mapping_eligible=True)
    return finalize(emitter)


def minified_transform(
    fixture_dir: Path,
    *,
    run_id: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    first = SourceDocument.read(fixture_dir / "adversarial-a.js")
    second = SourceDocument.read(fixture_dir / "adversarial-b.js")
    emitter = Emitter(run_id, "minified", "minified-output.js", "../src", observer, record_receipts)
    fragments = [
        (first, 'const café = "你好🔥";'),
        (second, 'const value = 7;'),
        (second, 'const label = "总计";'),
        (first, 'const combine = "é";'),
    ]
    for document, fragment in fragments:
        span = document.unique_span(fragment)
        collapsed = re.sub(r"\s+", " ", span.text).strip()
        emitter.emit(collapsed, operation_type="MINIFY_TO_SINGLE_LINE", origins=[span.configured(relation_role="source_map_anchor:MINIFY_TO_SINGLE_LINE")], mapping_eligible=True)
    _emit_synthetic(emitter, "console.log(café,value,label,combine);", "INSERT_LITERAL")
    return finalize(emitter)


def medium_transform(
    documents: dict[str, SourceDocument],
    *,
    run_id: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    emitter = Emitter(run_id, "medium", "medium-output.js", "../src", observer, record_receipts)
    for name in sorted(key for key in documents if key.startswith("medium-")):
        for span in documents[name].line_spans():
            if not span.text.strip():
                continue
            emitter.emit(span.text, operation_type="COPY_RANGE", origins=[span], mapping_eligible=True)
    return finalize(emitter)


def multistage_transform(
    fixture_dir: Path,
    *,
    run_id: str,
    stage1_observer: Callable[[dict], None] | None = None,
    stage2_observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> tuple[TransformResult, TransformResult]:
    documents = [
        SourceDocument.read(fixture_dir / "source-a.js"),
        SourceDocument.read(fixture_dir / "source-b.js"),
    ]
    stage1 = Emitter(run_id, "multistage_1", "bundled-intermediate.js", "../src", stage1_observer, record_receipts)
    for document in documents:
        for span in document.line_spans():
            if not span.text.strip():
                continue
            output = span.text.replace("export ", "", 1)
            stage1.emit(
                output,
                operation_type="CONCATENATE_SOURCES",
                origins=[span.configured(relation_role="source_map_anchor:CONCATENATE_SOURCES")],
                mapping_eligible=True,
            )
    first_result = finalize(stage1)
    intermediate = SourceDocument.from_text(first_result.generated_artifact, first_result.output_bytes.decode("utf-8"))
    support_by_start = {
        (row["generated_start"]["line"], row["generated_start"]["column"]): row["support_key"]
        for row in first_result.receipts if row["receipt_type"] == "emit"
    }
    if not support_by_start:
        support_by_start = {
            (span.start.line, span.start.column): f"multistage_1:support:{index:08d}"
            for index, span in enumerate(intermediate.line_spans()) if span.text.strip()
        }
    stage2 = Emitter(run_id, "multistage_2", "minified-final.js", "../src", stage2_observer, record_receipts)
    for span in intermediate.line_spans():
        if not span.text.strip():
            continue
        start_key = (span.start.line, span.start.column)
        prior_support = support_by_start[start_key]
        generated_origin = span.configured(
            origin_kind="generated_origin",
            prior_support_key=prior_support,
            relation_role="source_map_anchor:MINIFY_TO_SINGLE_LINE",
        )
        collapsed = re.sub(r"\s+", " ", span.text).strip()
        stage2.emit(
            collapsed,
            operation_type="MINIFY_TO_SINGLE_LINE",
            origins=[generated_origin],
            mapping_eligible=True,
        )
    return first_result, finalize(stage2)


def ambiguity_transform(
    source_path: Path,
    *,
    run_id: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    document = SourceDocument.read(source_path)
    emitter = Emitter(run_id, f"ambiguity_{source_path.stem}", "ambiguity-output.js", "../src", observer, record_receipts)
    fragment = next((line for line in document.text.splitlines() if line.strip()), None)
    if fragment is None:
        raise ValueError(f"AMBIGUITY_SOURCE_EMPTY:{source_path.name}")
    span = document.unique_span(fragment)
    emitter.emit(fragment + "\n", operation_type="COPY_RANGE", origins=[span], mapping_eligible=True)
    return finalize(emitter)


def equivalent_fact_transform(
    source_path: Path,
    *,
    run_id: str,
    strategy: str,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    document = SourceDocument.read(source_path)
    emitter = Emitter(run_id, f"fact_{strategy}", "fact-equivalent-output.js", "../src", observer, record_receipts)
    span = document.unique_span("const answer = 42;")
    if strategy == "direct_copy":
        output = span.text
        operation = "COPY_RANGE"
        parameters = {"strategy": "direct_copy"}
    elif strategy == "rewrite_then_restore":
        intermediate = span.text.replace("42", "41")
        output = intermediate.replace("41", "42")
        operation = "REWRITE_THEN_RESTORE"
        parameters = {"strategy": "rewrite_then_restore", "intermediate_sha256": __import__("hashlib").sha256(intermediate.encode()).hexdigest()}
    else:
        raise ValueError(f"STRATEGY_UNKNOWN:{strategy}")
    emitter.emit(
        output + "\n",
        operation_type=operation,
        origins=[span.configured(relation_role="source_map_anchor:COPY_RANGE")],
        mapping_eligible=True,
        transform_parameters=parameters,
    )
    return finalize(emitter)


def wide_relation_transform(
    fixture_dir: Path,
    *,
    run_id: str,
    include_wide_facts: bool,
    observer: Callable[[dict], None] | None = None,
    record_receipts: bool = True,
) -> TransformResult:
    first = SourceDocument.read(fixture_dir / "adversarial-a.js")
    second = SourceDocument.read(fixture_dir / "adversarial-b.js")
    emitter = Emitter(run_id, f"wide_{'rich' if include_wide_facts else 'narrow'}", "wide-equivalent-output.js", "../src", observer, record_receipts)
    anchor = first.unique_span("café").configured(
        original_name="café", generated_name="answer", relation_role="source_map_anchor:CONCATENATE_SOURCES"
    )
    origins = [anchor]
    if include_wide_facts:
        origins.append(second.unique_span("value").configured(
            mapping_anchor=False, relation_role="participation:secondary_actual_input"
        ))
    emitter.emit(
        "const answer = 42;\n",
        operation_type="CONCATENATE_SOURCES",
        origins=origins,
        mapping_eligible=True,
        transform_parameters={"wide_relation_enabled": include_wide_facts},
    )
    if include_wide_facts:
        emitter.dispose(first.unique_span("// DELETE_ME\n"), reason_code="wide_relation_deleted_input")
    return finalize(emitter)
