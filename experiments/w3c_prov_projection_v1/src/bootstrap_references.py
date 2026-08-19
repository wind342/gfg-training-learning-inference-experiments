from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


AUTHORITY_FILENAMES = {
    "PROV-DM": "prov-dm.html",
    "PROV-N": "prov-n.html",
    "PROV-O": "prov-o.html",
    "PROV-CONSTRAINTS": "prov-constraints.html",
    "PROV-ERRATA": "prov-errata.html",
    "PROV-IMPLEMENTATIONS": "prov-implementations.html",
    "PROV-CONSTRAINTS-TEST-PROCESS": "prov-constraints-test-process.html",
}
TEST_BASE = "https://dvcs.w3.org/hg/prov/raw-file/default/testcases/constraints/"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "source-information-continuity-w3c-prov-freezer/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _verified_write(path: Path, value: bytes, expected_sha256: str) -> None:
    digest = hashlib.sha256(value).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"frozen reference hash mismatch: {path.name}: {digest} != {expected_sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def bootstrap(root: Path) -> None:
    authority_manifest = json.loads((root / "authorities" / "w3c_prov_authority_manifest.json").read_text(encoding="utf-8"))
    authority_dir = root / "runtime" / "authorities"
    for document in authority_manifest["documents"]:
        name = AUTHORITY_FILENAMES[document["id"]]
        _verified_write(authority_dir / name, _download(document["url"]), document["sha256"])
        print(f"FROZEN_AUTHORITY={document['id']}", flush=True)
    official_results = json.loads((root / "artifacts" / "official_test_results.json").read_text(encoding="utf-8"))
    test_dir = root / "runtime" / "official_tests"
    for index, case in enumerate(official_results["cases"], start=1):
        _verified_write(test_dir / f"{case['id']}.provn", _download(TEST_BASE + case["id"] + ".provn"), case["sha256"])
        if index % 10 == 0 or index == len(official_results["cases"]):
            print(f"FROZEN_APPLICABLE_TEST_PROGRESS={index}/{len(official_results['cases'])}", flush=True)
    print(f"FROZEN_AUTHORITIES={len(authority_manifest['documents'])} FROZEN_APPLICABLE_TESTS={len(official_results['cases'])}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    bootstrap(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
