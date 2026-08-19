"""Download and decode the frozen PhysioNet WFDB record without a WFDB dependency."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = EXPERIMENT_ROOT / "contracts" / "input_manifest.json"
DEFAULT_DATA_ROOT = (
    EXPERIMENT_ROOT.parents[1]
    / "data_private"
    / "physionet"
    / "mitdb"
    / "1.0.0"
)


@dataclass(frozen=True)
class SignalWindow:
    record: str
    channel: str
    sample_rate_hz: int
    absolute_start: int
    digital_samples: np.ndarray
    physical_samples_mv: np.ndarray
    input_sha256: str


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_inputs(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    manifest = load_manifest()
    data_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for row in manifest["files"]:
        target = data_root / row["name"]
        if not target.exists():
            urllib.request.urlretrieve(row["url"], target)
        if target.stat().st_size != row["size"]:
            raise ValueError(f"INPUT_SIZE_MISMATCH:{row['name']}")
        if sha256_file(target) != row["sha256"]:
            raise ValueError(f"INPUT_HASH_MISMATCH:{row['name']}")
        result[row["name"]] = target
    return result


def decode_format_212(payload: bytes, signal_count: int = 2) -> np.ndarray:
    """Decode the two-channel WFDB 212 packing used by MIT-BIH record 100."""
    if signal_count != 2:
        raise ValueError("FORMAT_212_DECODER_REQUIRES_TWO_SIGNALS")
    if len(payload) % 3:
        raise ValueError("FORMAT_212_BYTE_COUNT_INVALID")
    triples = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
    first = triples[:, 0].astype(np.int16) + (
        (triples[:, 1] & 0x0F).astype(np.int16) << 8
    )
    second = triples[:, 2].astype(np.int16) + (
        (triples[:, 1] & 0xF0).astype(np.int16) << 4
    )
    first = np.where(first >= 2048, first - 4096, first).astype(np.int16)
    second = np.where(second >= 2048, second - 4096, second).astype(np.int16)
    return np.column_stack([first, second])


def _parse_header(header: str) -> dict:
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    record_fields = lines[0].split()
    signal_count = int(record_fields[1])
    result = {
        "record": record_fields[0],
        "signal_count": signal_count,
        "sample_rate_hz": int(record_fields[2]),
        "sample_count": int(record_fields[3]),
        "signals": [],
    }
    for line in lines[1 : 1 + signal_count]:
        fields = line.split()
        result["signals"].append(
            {
                "file": fields[0],
                "format": int(fields[1]),
                "gain": float(fields[2]),
                "adc_zero": int(fields[4]),
                "initial_value": int(fields[5]),
                "description": fields[8],
            }
        )
    return result


def load_signal_window(data_root: Path = DEFAULT_DATA_ROOT) -> SignalWindow:
    manifest = load_manifest()
    paths = materialize_inputs(data_root)
    header = _parse_header(paths["100.hea"].read_text(encoding="ascii"))
    if (
        header["record"] != manifest["record"]
        or header["signal_count"] != 2
        or header["sample_rate_hz"] != manifest["sample_rate_hz"]
        or any(row["format"] != 212 for row in header["signals"])
    ):
        raise ValueError("WFDB_HEADER_CONTRACT_MISMATCH")
    decoded = decode_format_212(paths["100.dat"].read_bytes())
    if decoded.shape != (header["sample_count"], header["signal_count"]):
        raise ValueError("WFDB_SAMPLE_COUNT_MISMATCH")
    channel_index = next(
        index
        for index, row in enumerate(header["signals"])
        if row["description"] == manifest["channel"]
    )
    signal = header["signals"][channel_index]
    if int(decoded[0, channel_index]) != signal["initial_value"]:
        raise ValueError("WFDB_INITIAL_VALUE_MISMATCH")
    start = manifest["window_start"]
    stop = start + manifest["window_length"]
    digital = decoded[start:stop, channel_index].astype(np.int16, copy=True)
    physical = (
        digital.astype(np.float64) - signal["adc_zero"]
    ) / signal["gain"]
    return SignalWindow(
        record=manifest["record"],
        channel=manifest["channel"],
        sample_rate_hz=manifest["sample_rate_hz"],
        absolute_start=start,
        digital_samples=digital,
        physical_samples_mv=physical,
        input_sha256=next(
            row["sha256"] for row in manifest["files"] if row["name"] == "100.dat"
        ),
    )

