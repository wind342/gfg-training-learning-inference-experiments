from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .coordinates import Position, index_to_position


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    logical_path: str
    text: str
    raw_bytes: bytes

    @classmethod
    def read(cls, path: Path, logical_path: str | None = None) -> "SourceDocument":
        raw = path.read_bytes()
        return cls(logical_path or path.name, raw.decode("utf-8"), raw)

    @classmethod
    def from_text(cls, logical_path: str, text: str) -> "SourceDocument":
        raw = text.encode("utf-8")
        return cls(logical_path, text, raw)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.raw_bytes)

    def unique_span(self, fragment: str, *, occurrence: int = 0) -> "OriginSpan":
        starts: list[int] = []
        cursor = 0
        while True:
            found = self.text.find(fragment, cursor)
            if found < 0:
                break
            starts.append(found)
            cursor = found + max(1, len(fragment))
        if occurrence < 0 or occurrence >= len(starts):
            raise ValueError(f"SOURCE_FRAGMENT_NOT_FOUND:{self.logical_path}:{fragment!r}:{occurrence}")
        start = starts[occurrence]
        return OriginSpan(self, start, start + len(fragment))

    def line_spans(self) -> list["OriginSpan"]:
        rows: list[OriginSpan] = []
        cursor = 0
        for line in self.text.splitlines(keepends=True):
            end = cursor + len(line)
            rows.append(OriginSpan(self, cursor, end))
            cursor = end
        if cursor < len(self.text):
            rows.append(OriginSpan(self, cursor, len(self.text)))
        return rows


@dataclass(frozen=True)
class OriginSpan:
    document: SourceDocument
    start_index: int
    end_index: int
    origin_kind: str = "registered_source"
    prior_support_key: str | None = None
    original_name: str | None = None
    generated_name: str | None = None
    mapping_anchor: bool = True
    relation_role: str = "source_map_anchor:COPY_RANGE"

    @property
    def text(self) -> str:
        return self.document.text[self.start_index:self.end_index]

    @property
    def start(self) -> Position:
        return index_to_position(self.document.text, self.start_index)

    @property
    def end(self) -> Position:
        return index_to_position(self.document.text, self.end_index)

    def configured(
        self,
        *,
        origin_kind: str | None = None,
        prior_support_key: str | None = None,
        original_name: str | None = None,
        generated_name: str | None = None,
        mapping_anchor: bool | None = None,
        relation_role: str | None = None,
    ) -> "OriginSpan":
        return OriginSpan(
            self.document,
            self.start_index,
            self.end_index,
            origin_kind=origin_kind or self.origin_kind,
            prior_support_key=prior_support_key if prior_support_key is not None else self.prior_support_key,
            original_name=original_name if original_name is not None else self.original_name,
            generated_name=generated_name if generated_name is not None else self.generated_name,
            mapping_anchor=self.mapping_anchor if mapping_anchor is None else mapping_anchor,
            relation_role=relation_role or self.relation_role,
        )

    def as_receipt_origin(self) -> dict:
        return {
            "origin_kind": self.origin_kind,
            "source_file": self.document.logical_path,
            "source_artifact_sha256": self.document.sha256,
            "source_start": self.start.as_dict(),
            "source_end": self.end.as_dict(),
            "source_bytes_sha256": sha256_bytes(self.text.encode("utf-8")),
            "source_text": self.text,
            "source_content": self.document.text,
            "original_name": self.original_name,
            "generated_name": self.generated_name,
            "mapping_anchor": self.mapping_anchor,
            "relation_role": self.relation_role,
            "prior_support_key": self.prior_support_key,
        }


ReceiptObserver = Callable[[dict], None]


@dataclass
class Emitter:
    run_id: str
    stage_id: str
    generated_artifact: str
    source_root: str | None = None
    observer: ReceiptObserver | None = None
    record_receipts: bool = True
    _chunks: list[str] = field(default_factory=list)
    _receipts: list[dict] = field(default_factory=list)
    _ordinal: int = 0

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    @property
    def receipts(self) -> tuple[dict, ...]:
        return tuple(self._receipts)

    def emit(
        self,
        text: str,
        *,
        operation_type: str,
        origins: list[OriginSpan],
        mapping_eligible: bool,
        transform_parameters: dict | None = None,
    ) -> dict:
        if not text:
            raise ValueError("EMPTY_EMIT_PROHIBITED")
        if not origins:
            raise ValueError("ORIGIN_REQUIRED")
        anchors = sum(origin.mapping_anchor for origin in origins)
        if mapping_eligible and anchors != 1:
            raise ValueError("MAPPED_EMIT_REQUIRES_ONE_ANCHOR")
        if not mapping_eligible and anchors:
            raise ValueError("UNMAPPED_EMIT_REQUIRES_ZERO_ANCHORS")
        start = index_to_position(self.text, len(self.text))
        self._chunks.append(text)
        end = index_to_position(self.text, len(self.text))
        support_key = f"{self.stage_id}:support:{self._ordinal:08d}"
        receipt = {
            "receipt_type": "emit",
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "occurrence_key": f"{self.stage_id}:occurrence:{self._ordinal:08d}",
            "occurrence_index": self._ordinal,
            "operation_type": operation_type,
            "transform_parameters": transform_parameters or {},
            "generated_artifact": self.generated_artifact,
            "source_root": self.source_root,
            "generated_start": start.as_dict(),
            "generated_end": end.as_dict(),
            "generated_bytes_sha256": sha256_bytes(text.encode("utf-8")),
            "generated_text": text,
            "support_key": support_key,
            "mapping_eligible": mapping_eligible,
            "origins": [origin.as_receipt_origin() for origin in origins],
        }
        if self.record_receipts:
            self._receipts.append(receipt)
            if self.observer is not None:
                self.observer(receipt)
        self._ordinal += 1
        return receipt

    def dispose(
        self,
        origin: OriginSpan,
        *,
        operation_type: str = "DELETE_RANGE",
        reason_code: str = "deleted_by_frozen_transform",
        transform_parameters: dict | None = None,
    ) -> dict:
        receipt = {
            "receipt_type": "disposition",
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "occurrence_key": f"{self.stage_id}:occurrence:{self._ordinal:08d}",
            "occurrence_index": self._ordinal,
            "operation_type": operation_type,
            "transform_parameters": transform_parameters or {},
            "generated_artifact": self.generated_artifact,
            "source_root": self.source_root,
            "disposition_reason_code": reason_code,
            "origins": [origin.configured(mapping_anchor=False, relation_role="participation:deleted_range").as_receipt_origin()],
        }
        if self.record_receipts:
            self._receipts.append(receipt)
            if self.observer is not None:
                self.observer(receipt)
        self._ordinal += 1
        return receipt


@dataclass(frozen=True)
class TransformResult:
    run_id: str
    stage_id: str
    generated_artifact: str
    output_bytes: bytes
    output_sha256: str
    receipts: tuple[dict, ...]


def finalize(emitter: Emitter) -> TransformResult:
    output = emitter.text.encode("utf-8")
    return TransformResult(
        run_id=emitter.run_id,
        stage_id=emitter.stage_id,
        generated_artifact=emitter.generated_artifact,
        output_bytes=output,
        output_sha256=sha256_bytes(output),
        receipts=emitter.receipts,
    )
