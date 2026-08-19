from __future__ import annotations

import json
import re
from typing import Any

from .record_model import sorted_records


_HEADER = (
    "document\n"
    "  prefix ex <https://example.org/w3c-prov-projection-v1#>\n"
    "  prefix prov <http://www.w3.org/ns/prov#>\n"
    "  prefix xsd <http://www.w3.org/2001/XMLSchema#>\n"
)


def _attribute_text(types: list[str], attributes: dict[str, Any]) -> str:
    values = [f"prov:type='{item}'" for item in sorted(types)]
    for key, value in sorted(attributes.items()):
        if isinstance(value, int):
            values.append(f'{key}="{value}" %% xsd:integer')
        else:
            values.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return "[" + ", ".join(values) + "]"


def serialize_provn(records: list[dict[str, Any]]) -> bytes:
    lines: list[str] = []
    for record in sorted_records(records):
        kind = record["kind"]
        if kind == "entity":
            lines.append(f"  entity({record['id']}, {_attribute_text(record['types'], record['attributes'])})")
        elif kind == "activity":
            lines.append(f"  activity({record['id']}, -, -, {_attribute_text(record['types'], record['attributes'])})")
        elif kind == "agent":
            lines.append(f"  agent({record['id']}, {_attribute_text(record['types'], record['attributes'])})")
        elif kind == "usage":
            lines.append(
                f"  used({record['id']}; {record['activity']}, {record['entity']}, -, "
                f"[prov:role='{record['role']}', ex:relationOrdinal=\"{record['ordinal']}\" %% xsd:integer])"
            )
        elif kind == "generation":
            lines.append(f"  wasGeneratedBy({record['id']}; {record['entity']}, {record['activity']}, -)")
        elif kind == "derivation":
            lines.append(
                f"  wasDerivedFrom({record['id']}; {record['generated_entity']}, {record['used_entity']}, "
                f"{record['activity']}, {record['generation']}, {record['usage']}, "
                f"[prov:role='{record['role']}', ex:relationOrdinal=\"{record['ordinal']}\" %% xsd:integer])"
            )
        elif kind == "association":
            lines.append(
                f"  wasAssociatedWith({record['id']}; {record['activity']}, {record['agent']}, -, "
                f"[prov:role='{record['role']}', ex:relationOrdinal=\"{record['ordinal']}\" %% xsd:integer])"
            )
        else:
            raise ValueError(f"unknown normalized record kind: {kind}")
    return (_HEADER + "\n".join(lines) + "\nendDocument\n").encode("utf-8")


def _parse_attributes(text: str) -> tuple[list[str], dict[str, Any]]:
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"malformed PROV-N attributes: {text}")
    types: list[str] = []
    attributes: dict[str, Any] = {}
    body = text[1:-1]
    if not body:
        return types, attributes
    for item in body.split(", "):
        key, raw = item.split("=", 1)
        if key == "prov:type":
            if not (raw.startswith("'") and raw.endswith("'")):
                raise ValueError("prov:type must be a qualified name")
            types.append(raw[1:-1])
        elif raw.endswith(" %% xsd:integer"):
            attributes[key] = int(json.loads(raw.removesuffix(" %% xsd:integer")))
        else:
            attributes[key] = json.loads(raw)
    return sorted(types), dict(sorted(attributes.items()))


def _relation_attrs(text: str) -> tuple[str, int]:
    match = re.fullmatch(
        r"\[prov:role='(ex:[A-Za-z_][A-Za-z0-9_]*)', ex:relationOrdinal=\"([0-9]+)\" %% xsd:integer\]",
        text,
    )
    if match is None:
        raise ValueError(f"malformed relation attributes: {text}")
    return match.group(1), int(match.group(2))


def parse_provn(document: bytes) -> list[dict[str, Any]]:
    text = document.decode("utf-8")
    lines = text.splitlines()
    if lines[:4] != _HEADER.rstrip("\n").splitlines() or not lines or lines[-1] != "endDocument":
        raise ValueError("malformed deterministic PROV-N document wrapper")
    records: list[dict[str, Any]] = []
    qname = r"ex:[A-Za-z_][A-Za-z0-9_]*"
    for line in lines[4:-1]:
        if not line.startswith("  "):
            raise ValueError(f"malformed PROV-N statement indentation: {line}")
        statement = line[2:]
        match = re.fullmatch(rf"entity\(({qname}), (\[.*\])\)", statement)
        if match:
            types, attributes = _parse_attributes(match.group(2))
            records.append({"kind": "entity", "id": match.group(1), "types": types, "attributes": attributes})
            continue
        match = re.fullmatch(rf"activity\(({qname}), -, -, (\[.*\])\)", statement)
        if match:
            types, attributes = _parse_attributes(match.group(2))
            records.append({"kind": "activity", "id": match.group(1), "types": types, "attributes": attributes})
            continue
        match = re.fullmatch(rf"agent\(({qname}), (\[.*\])\)", statement)
        if match:
            types, attributes = _parse_attributes(match.group(2))
            records.append({"kind": "agent", "id": match.group(1), "types": types, "attributes": attributes})
            continue
        match = re.fullmatch(rf"used\(({qname}); ({qname}), ({qname}), -, (\[.*\])\)", statement)
        if match:
            role, ordinal = _relation_attrs(match.group(4))
            records.append({"kind": "usage", "id": match.group(1), "activity": match.group(2), "entity": match.group(3), "role": role, "ordinal": ordinal})
            continue
        match = re.fullmatch(rf"wasGeneratedBy\(({qname}); ({qname}), ({qname}), -\)", statement)
        if match:
            records.append({"kind": "generation", "id": match.group(1), "entity": match.group(2), "activity": match.group(3)})
            continue
        match = re.fullmatch(rf"wasDerivedFrom\(({qname}); ({qname}), ({qname}), ({qname}), ({qname}), ({qname}), (\[.*\])\)", statement)
        if match:
            role, ordinal = _relation_attrs(match.group(7))
            records.append({
                "kind": "derivation", "id": match.group(1), "generated_entity": match.group(2),
                "used_entity": match.group(3), "activity": match.group(4), "generation": match.group(5),
                "usage": match.group(6), "role": role, "ordinal": ordinal,
            })
            continue
        match = re.fullmatch(rf"wasAssociatedWith\(({qname}); ({qname}), ({qname}), -, (\[.*\])\)", statement)
        if match:
            role, ordinal = _relation_attrs(match.group(4))
            records.append({"kind": "association", "id": match.group(1), "activity": match.group(2), "agent": match.group(3), "role": role, "ordinal": ordinal})
            continue
        raise ValueError(f"unsupported or malformed PROV-N statement: {statement}")
    return sorted_records(records)
