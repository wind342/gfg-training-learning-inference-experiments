# GFG-driven nanoGPT mechanism-discovery archive

This directory is the durable, cumulative archive for AI answers produced by
the nanoGPT mechanism-discovery experiments and for the exact participant GFG
that each answer consumed.

The archive deliberately separates two storage layers:

- `entries/` and `archive_index.json` are Git-tracked. They preserve each
  submitted report and executable candidate, its frozen task contract,
  candidate/session evidence, all available evaluation revisions, and a
  content-addressed commitment to the corresponding GFG.
- `data_private/gfg_nanogpt_mechanism_discovery_archive/gfg/<bundle-id>/` is
  inside the same local repository but excluded from Git. It contains the
  exact participant GFG bytes. These bundles are about 1 GiB each and include
  SQLite databases larger than GitHub's ordinary file limit. On the original
  machine they are preserved with hard links when possible, so preservation
  does not duplicate their disk blocks.

The tracked `gfg_manifest.json` for every entry records the participant bundle
identity, database SHA-256, validation identity and status, tensor-object
count, content-addressed path commitment, graph counts, logical byte size and
the relative local archive path. Thus a missing private bundle is visible; it
cannot be silently treated as archived evidence.

## Status discipline

Every discovered answer is retained. A sealed candidate remains `SEALED` even
when its scientific evaluation fails. A platform-aborted draft is retained as
`UNSEALED_PLATFORM_ABORTED_SUPERSEDED` and is never counted as a formal
candidate. Runtime repairs create additional immutable evaluation revisions;
they do not replace or edit the submitted answer.

## Commands

From the repository root:

```text
python scripts/archive_nanogpt_discovery.py --repository-root . import-all --runs-root <runs-root>
python scripts/archive_nanogpt_discovery.py --repository-root . verify
python scripts/archive_nanogpt_discovery.py --repository-root . verify --deep
```

`verify` hashes every tracked submission. `verify --deep` additionally hashes
each archived GFG database and every content-addressed tensor object; it is
intentionally slower.

Completed future formal instances are archived automatically by the formal
runner. The standalone command remains available for recovered, repaired or
historical instances.
