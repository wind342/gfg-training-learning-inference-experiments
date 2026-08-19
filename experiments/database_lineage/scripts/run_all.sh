#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON:-python3}"
mode="${1:---full}"

cd "${repo_root}"
exec "${python_bin}" experiments/database_lineage/scripts/run_all.py "${mode}"
