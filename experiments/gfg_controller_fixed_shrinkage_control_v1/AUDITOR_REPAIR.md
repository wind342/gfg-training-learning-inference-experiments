# Publication-boundary audit repair (no scientific change)

The initial full independent audit passed before git staging (AUDIT.json).
Re-executing the same checker after staging the new experiment produced:

```
File "audit.py", line 19, in main
    assert not run.git('diff',run.BASE,'--name-only'), 'OLD_TRACKED_FILES_CHANGED'
AssertionError: OLD_TRACKED_FILES_CHANGED
```

Cause: this Git check incorrectly included newly added files as changes to old
tracked files. `git diff --cached --name-status` showed ONLY twelve A entries,
all under this new experiment directory. No old file was modified.

The repair excludes A (added) statuses, retaining C/D/M/R/T/U/X/B checks. All
baseline Python hashes remain checked. The old auditor is preserved verbatim as
audit_prepublication.py.snapshot, matching its original pre-run seal hash.
RUN_SEAL.json, the original AUDIT.json, all scientific sources, PROTOCOL.md,
SELECTION.json, RESULTS.json and all native artifacts are unchanged.

AUDITOR_REVISION_SEAL.json binds the original seal and original/revised auditor
hashes BEFORE rerunning the corrected checker. This seal is explicitly post-run:
it is NOT represented as a pre-outcome scientific protocol. Only the publication
Git boundary and audit-receipt preservation handling changed. The corrected
checker independently reprocesses every artifact and writes AUDIT_PUBLICATION.json.
Subsequent checks write a uniquely named AUDIT_RECHECK receipt, never overwrite.
