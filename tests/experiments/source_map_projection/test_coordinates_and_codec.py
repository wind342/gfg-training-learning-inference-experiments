from __future__ import annotations

import pytest

from experiments.source_map_projection.src.canonical_source_map import (
    decode_source_map,
    encode_source_map,
)
from experiments.source_map_projection.src.coordinates import (
    CoordinateError,
    Position,
    index_to_position,
    position_to_index,
    utf16_units,
    validate_round_trip,
)


def test_utf16_and_crlf_coordinates_round_trip() -> None:
    text = "ASCII 中文 🔥 e\u0301\t\r\n\r\nlast\n"
    validate_round_trip(text)
    assert utf16_units("🔥") == 2
    assert index_to_position("a🔥b", 2) == Position(0, 3)
    assert index_to_position("a\r\nb", 3) == Position(1, 0)
    assert position_to_index("a\r\nb", Position(1, 0)) == 3
    with pytest.raises(CoordinateError, match="POSITION_INSIDE_UTF16_SURROGATE_PAIR"):
        position_to_index("🔥", Position(0, 1))
    with pytest.raises(CoordinateError, match="POSITION_INSIDE_CRLF"):
        index_to_position("a\r\nb", 2)


def test_ordinary_map_codec_preserves_mapped_unmapped_names_and_sources() -> None:
    records = [
        {
            "generated_file": "out.js", "generated_line": 0, "generated_column": 0,
            "mapped": False, "original_source": None, "original_line": None,
            "original_column": None, "original_name": None,
        },
        {
            "generated_file": "out.js", "generated_line": 1, "generated_column": 3,
            "mapped": True, "original_source": "中文.js", "original_line": 2,
            "original_column": 4, "original_name": "火🔥",
        },
    ]
    document = encode_source_map(
        records, generated_file="out.js", source_root="../src",
        source_contents={"中文.js": "const 火🔥 = 1;\n"},
    )
    decoded = decode_source_map(document, base_url="file:///maps/out.js.map", require_strict_order=True)
    normalized = [{key: value for key, value in row.items() if key != "resolved_original_source"} for row in decoded.records]
    assert normalized == records
    assert decoded.records[1]["resolved_original_source"] == "file:///src/中文.js"
