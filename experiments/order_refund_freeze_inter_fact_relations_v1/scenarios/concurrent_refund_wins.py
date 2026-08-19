from ..src.orchestrator import run_workflow


def run(*, repeat_index: int = 1, capture_enabled: bool = True):
    return run_workflow(
        "CONCURRENT_REFUND_WINS",
        repeat_index=repeat_index,
        capture_enabled=capture_enabled,
    )
