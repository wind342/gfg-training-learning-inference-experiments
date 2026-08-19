from ..src.orchestrator import run_workflow


def run(*, repeat_index: int = 1, capture_enabled: bool = True):
    return run_workflow(
        "LATE_REFUND_AFTER_FREEZE",
        repeat_index=repeat_index,
        capture_enabled=capture_enabled,
    )
