from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_profile_and_crosswalk_are_frozen_and_closed() -> None:
    profile = json.loads((ROOT / "profiles" / "w3c_prov_generation_profile_v1.json").read_text(encoding="utf-8"))
    crosswalk = json.loads((ROOT / "profiles" / "core_to_w3c_prov_crosswalk_v1.json").read_text(encoding="utf-8"))
    assert profile["profile_id"] == "w3c-prov-generation-profile-v1"
    assert profile["status"] == "FROZEN_BEFORE_IMPLEMENTATION"
    assert crosswalk["status"] == "FROZEN_BEFORE_IMPLEMENTATION"
    assert "complete Snapshot" in profile["prohibited_attributes"]
    assert "EvidenceRecord" in crosswalk["excluded_core_types"]


def test_authority_files_are_manifests_not_downloaded_documents() -> None:
    authority_files = list((ROOT / "authorities").iterdir())
    assert authority_files
    assert all(path.suffix == ".json" for path in authority_files)
    assert max(path.stat().st_size for path in authority_files) < 100_000

