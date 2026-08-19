from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPOSITORY_ROOT / "src"
for path in (str(REPOSITORY_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.inter_fact_relations_v0_hardening_scale_v1.src.materialize import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    sys.exit(main())
