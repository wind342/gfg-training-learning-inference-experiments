from __future__ import annotations

import re
import subprocess
from pathlib import Path

from experiments.five_profile_unified_projection_proof.src.manifest import (
    PROFILE_DOCUMENTS,
    _core_changed_paths,
    _document_hashes,
)
from experiments.five_profile_unified_projection_proof.src.mechanism_entry import CORE_COMMIT, SOURCE_COMMITS


REPO = Path(__file__).resolve().parents[3]


def test_manifest_profile_and_crosswalk_hashes_cover_five_mechanisms():
    hashes = _document_hashes(REPO)
    assert set(hashes) == set(PROFILE_DOCUMENTS)
    for documents in hashes.values():
        for key, row in documents.items():
            if key != "crosswalk_location":
                assert (REPO / row["path"]).is_file()
                assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])


def test_every_source_commit_and_core_commit_exists():
    for commit in (*SOURCE_COMMITS.values(), CORE_COMMIT):
        process = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO)
        assert process.returncode == 0


def test_unified_delivery_changes_zero_core_files():
    assert _core_changed_paths(REPO) == []

