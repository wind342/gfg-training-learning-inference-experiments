# Order-refund-freeze experiment design

The orchestrator initializes a real SQLite file in WAL mode and coordinates
separate refund, freeze, and notification processes with explicit barriers
and events. Refund messages cross an actual multiprocessing queue.

Each worker returns raw execution receipts. Subsequent collectors may create
atomic five-coordinate facts and an external relation sidecar, but they do
not decide whether a transaction committed. Candidate, reference, native
trace, and compare resolvers run with disjoint serialized inputs in separate
processes.

Capture-enabled and capture-disabled paired executions retain identical
business inputs and schedules. Relation metadata is excluded from ordinary
business output.
