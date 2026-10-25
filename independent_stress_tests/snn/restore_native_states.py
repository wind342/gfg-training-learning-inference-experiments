"""Reconstruct exact SNN PyTorch state files from repository-native binary Git blobs.

The four replay state files are byte-identical to the four originals. This
script recreates both without duplicating 16 MiB in Git. No pickle is loaded.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    '71001': 'bcec4e7dc5142844e1186e2ac262abc582a9ba9210dda8ad5e888d01f1989436',
    '71002': 'be331d57bed6f6d5f9d98fe20b3f98b6d3a2f73f96591c91bb3813d1655b802a',
    '71003': '59bf75e251af210a3208626149d356be2072fe7974e46da0b4ca7fc424aeb564',
    '71004': '9b3d1899fa5a34d1dfe3e8094f8231b9deb74c542df9972d6346f8ffd8345330'
}
PART_COUNT = 4
BYTES_PER_SEED = 3968374
BASENAME = "NATIVE_STATES_AND_OUTCOMES.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str) -> None:
    assert path.stat().st_size == BYTES_PER_SEED, f"size mismatch: {path}"
    actual = sha256_file(path)
    assert actual == expected, f"SHA-256 mismatch {path}: {actual}"


def main() -> None:
    for seed, expected in EXPECTED.items():
        parts = [ROOT / "native_parts" / f"seed_{seed}" / f"part_{i:03d}.bin" for i in range(PART_COUNT)]
        for part in parts:
            if not part.is_file():
                raise FileNotFoundError(f"missing evidence fragment: {part}")
        formal = ROOT / "formal" / f"seed_{seed}" / BASENAME
        formal.parent.mkdir(parents=True, exist_ok=True)
        if not formal.exists():
            pending = formal.with_name(BASENAME + ".assembling")
            with pending.open("xb") as out:
                for part in parts:
                    with part.open("rb") as fh:
                        shutil.copyfileobj(fh, out, length=1024 * 1024)
            verify(pending, expected)
            pending.replace(formal)
        verify(formal, expected)
        replay = ROOT / "replay" / f"seed_{seed}" / BASENAME
        replay.parent.mkdir(parents=True, exist_ok=True)
        if not replay.exists():
            shutil.copyfile(formal, replay)
        verify(replay, expected)
        print(f"{seed}: original/replay verified {expected}")
    print("All four SNN seeds restored and SHA-256 verified.")


if __name__ == "__main__":
    main()
