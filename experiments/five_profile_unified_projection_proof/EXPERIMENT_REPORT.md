# Five frozen mechanism projection proofs: unified reproduction

Final machine status: `FIVE_PROFILE_EXACT_STRICT_PROJECTION_SUPPORTED`.

Two complete executions produced the same canonical summary SHA-256:
`6d123156ffe4a206e47ce9c27a11dadfd506add5796d7537a11ffe9d2f4dfbfe`.
Each execution also passed 204/204 tests with zero failures, errors, or skips.

## Structural unity

All five candidate paths project from the frozen Core relation structure:
typed origin, `GenerationOccurrence`, typed outcome, relation role,
`GenerationBinding`, and `GeneratedOrigin`. Domain adapters determine their
declared profile payloads; they do not add mechanism-specific fields to Core.

## P1 and P2

| Mechanism | P1 | P2 | Comparison scale | Mismatches |
|---|---:|---:|---:|---:|
| Database which-lineage | PASS, 112/112 | PASS, 2/2 witnesses | 112 canonical records | 0 |
| ECMA-426 ordinary Source Map | PASS, 685/685 | PASS, 3/3 witnesses | 685 segments; 1,385 queries | 0 |
| OpenTelemetry occurrence trace | PASS, 61,368/61,368 | PASS, 2/2 witnesses | 61,368 spans; 2,382 links | 0 |
| W3C PROV generation profile | PASS, 51/51 | PASS, 4/4 witnesses | 51 normalized records | 0 |
| PyTorch Autograd dependency profile | PASS, 33 nodes + 33 edges | PASS, 3/3 witnesses | 29/29 hardened dependencies | 0 |

Source Map multistage composition preserved five `GeneratedOrigin` bridges and
created zero direct original-to-final shortcuts. OpenTelemetry direct Core,
native SDK, and Core-to-Database-to-OTel traces were exact. W3C included the
real left- versus right-associative transform witness. Autograd preserved an
equal native graph for the divergent checkpoint pair while complete facts and
gradients differed.

## External independence

External independence is a separate evidence-strength dimension: Database C,
Source Map B, OpenTelemetry B, W3C PROV C, and PyTorch Autograd A. These ratings
do not change the constructive P1/P2 truth values.

## Core baseline reconciliation

The imported Autograd proof pins a protected Core tree older than the current
audited `6b34906d7b6e4fa15f6c7d6e3013daa35a308b5e` baseline. The later Core
hardening changes content-addressed snapshot identities, so exactly five legacy
snapshot/identity-bearing v1 artifact baselines required regeneration from a
current real execution. The unified runner does not weaken scientific comparisons: it requires all
current 33-node/33-edge projections, all three P2 witnesses, all queries and
checkpoint semantics, every non-legacy hardening component, and the hardened
29-relation native intervention comparison to be exact. It separately requires
zero Core file changes relative to the current 6b3490 baseline.

The unified manifest independently rehashed all 17 non-manifest evidence
files with zero failures. The unified package changes zero protected Core
files.

## Scope and non-claims

Within five explicitly frozen and limited profiles/workloads, complete target
representations are exactly equal and valid non-injectivity witnesses exist.
This is not whole-standard coverage, arbitrary-program coverage, a uniqueness
or minimality claim for Core, a unique-crosswalk claim, or a claim that all five
mechanisms have equal third-party independence.

## Unified reproduction

W3C and Autograd previously lived on independent experiment submissions. This
package is the first checkout in this repository lineage that imports both and
actually replays all five proofs through one structured, fail-closed entrypoint.
