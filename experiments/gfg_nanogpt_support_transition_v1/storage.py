from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    require,
)


def raw_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().cpu().numpy()
    return np.ascontiguousarray(value)


class TensorStore:
    """Content-addressed external tensor payloads for transition evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        value: torch.Tensor | np.ndarray,
        *,
        representation: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        array = raw_array(value)
        raw_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        path = self.root / f"{raw_sha}.npy"
        if not path.exists():
            temporary = self.root / f".{raw_sha}.npy.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            temporary_file_sha = file_sha256(temporary)
            temporary_loaded = np.load(temporary, allow_pickle=False, mmap_mode="r")
            require(
                hashlib.sha256(np.ascontiguousarray(temporary_loaded).tobytes(order="C")).hexdigest()
                == raw_sha,
                "CST_TENSOR_TEMP_RAW_HASH_MISMATCH",
            )
            del temporary_loaded
            temporary.replace(path)
            require(file_sha256(path) == temporary_file_sha, "CST_TENSOR_ATOMIC_REPLACE_HASH_MISMATCH")
        loaded = np.load(path, allow_pickle=False, mmap_mode="r")
        require(
            hashlib.sha256(np.ascontiguousarray(loaded).tobytes(order="C")).hexdigest()
            == raw_sha,
            "CST_TENSOR_STORE_RAW_HASH_MISMATCH",
        )
        return {
            "dtype": str(array.dtype),
            "file_sha256": file_sha256(path),
            "locator": f"tensor-objects/{path.name}",
            "raw_tensor_sha256": raw_sha,
            "representation": representation,
            "shape": list(array.shape),
            **dict(extra or {}),
        }

    def put_named(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        representation: str,
    ) -> dict[str, Any]:
        names = sorted(values)
        require(bool(names), "CST_NAMED_TENSOR_SET_EMPTY")
        arrays = [raw_array(values[name]).reshape(-1) for name in names]
        dtype = np.result_type(*[array.dtype for array in arrays])
        packed = np.concatenate([array.astype(dtype, copy=False) for array in arrays])
        offset = 0
        layout: list[dict[str, Any]] = []
        for name, source, flat in zip(names, (raw_array(values[name]) for name in names), arrays):
            count = int(flat.size)
            layout.append(
                {
                    "dtype": str(source.dtype),
                    "element_count": count,
                    "name": name,
                    "offset": offset,
                    "raw_tensor_sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
                    "shape": list(source.shape),
                }
            )
            offset += count
        return self.put(
            packed,
            representation=representation,
            extra={"canonical_name_order": names, "layout": layout},
        )
