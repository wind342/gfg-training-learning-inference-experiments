from __future__ import annotations

from dataclasses import dataclass


class CoordinateError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Position:
    line: int
    column: int

    def as_dict(self) -> dict[str, int]:
        return {"line": self.line, "column": self.column}


def utf16_units(value: str) -> int:
    """Return the ECMA-426 JavaScript column width of a string."""
    return len(value.encode("utf-16-le")) // 2


def index_to_position(text: str, index: int) -> Position:
    if index < 0 or index > len(text):
        raise CoordinateError("INDEX_OUT_OF_RANGE")
    line = 0
    column = 0
    cursor = 0
    while cursor < index:
        char = text[cursor]
        if char == "\r":
            if cursor + 1 < len(text) and text[cursor + 1] == "\n":
                if cursor + 1 >= index:
                    raise CoordinateError("POSITION_INSIDE_CRLF")
                cursor += 2
            else:
                cursor += 1
            line += 1
            column = 0
            continue
        if char == "\n":
            cursor += 1
            line += 1
            column = 0
            continue
        column += utf16_units(char)
        cursor += 1
    return Position(line, column)


def position_to_index(text: str, position: Position) -> int:
    if position.line < 0 or position.column < 0:
        raise CoordinateError("POSITION_NEGATIVE")
    line = 0
    column = 0
    cursor = 0
    while cursor < len(text):
        if line == position.line and column == position.column:
            return cursor
        char = text[cursor]
        if char == "\r":
            cursor += 2 if cursor + 1 < len(text) and text[cursor + 1] == "\n" else 1
            line += 1
            column = 0
            continue
        if char == "\n":
            cursor += 1
            line += 1
            column = 0
            continue
        width = utf16_units(char)
        if line == position.line and column < position.column < column + width:
            raise CoordinateError("POSITION_INSIDE_UTF16_SURROGATE_PAIR")
        column += width
        cursor += 1
    if line == position.line and column == position.column:
        return cursor
    raise CoordinateError("POSITION_OUT_OF_RANGE")


def validate_round_trip(text: str) -> None:
    for index in range(len(text) + 1):
        if index and text[index - 1:index + 1] == "\r\n":
            continue
        position = index_to_position(text, index)
        if position_to_index(text, position) != index:
            raise CoordinateError("COORDINATE_ROUND_TRIP_FAILED")
