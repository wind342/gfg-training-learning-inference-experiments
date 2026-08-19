from pathlib import Path

from .independent import check


if __name__ == "__main__":
    check(Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "support_reallocation_audit_v1")
