# FA-E02 — Existing Evidence Audit

## Frozen scope

This audit reads the already executed nanoGPT inference GFGs only. It performs no training, no new inference, and no triple- or four-component gate. FA-E01 and its `INCONCLUSIVE` verdict are not modified.

- Source histories: **13**
- Held-out instances per history: **212**
- Historical phases: **pre-formation, formed, decline, recovered**
- Native conditions per instance: **11** (baseline + 4 single gates + 6 pair gates)
- New native executions: **0**

## Scientific object

The unit of evidence is not a common geometric fingerprint. It is:

`support configuration -> native causal intervention -> target-boundary consequence`

A component is called *contextually necessary* only when the baseline target is correct and its single gate changes that exact target to incorrect. A pair is *backup/substitution-compatible* only when both relevant single gates retain the correct target but their joint gate crosses the target boundary. These terms are restricted to the executed query, target, receiving state, component versions, and gate site.

## What the existing evidence directly adjudicates

At the formed phase, the native evidence contains **2618** baseline-correct instance observations across all runs. **2381** of them (**90.95%**) remain correct under more than one of the 11 observed configurations, establishing observed causal multiplicity without requiring the configurations to be geometrically similar.

### Contextual necessity by component (formed phase)

| Component | Boundary-crossing single-gate cases |
|---|---:|
| `h0.attn` | 2488 |
| `h0.mlp` | 2221 |
| `h1.attn` | 593 |
| `h1.mlp` | 1360 |

### Backup/substitution-compatible pair cases (formed phase)

| Gated pair | Singles individually retain target, pair loses target |
|---|---:|
| `h0.attn+h0.mlp` | 11 |
| `h0.attn+h1.attn` | 40 |
| `h0.attn+h1.mlp` | 39 |
| `h0.mlp+h1.attn` | 83 |
| `h0.mlp+h1.mlp` | 119 |
| `h1.attn+h1.mlp` | 254 |

Across the six pairs this criterion is met in **546** formed-phase instance-pair cases.

### Pair non-additivity (formed phase)

| Gated pair | Instance interactions with `abs(I) > 1e-6` |
|---|---:|
| `h0.attn+h0.mlp` | 2756 |
| `h0.attn+h1.attn` | 2756 |
| `h0.attn+h1.mlp` | 2756 |
| `h0.mlp+h1.attn` | 2756 |
| `h0.mlp+h1.mlp` | 2756 |
| `h1.attn+h1.mlp` | 2756 |

Target-conditioned boundary consequences vary across target groups in **52** of the formed run-by-component cases with observable baseline-correct targets. This supports shared plus target-conditioned organization rather than a fixed single owner.

## Historical reorganization

The same 212 held-out positions are aligned within each run across phases. The table reports whether their 10-condition boundary-consequence pattern changes while the baseline target remains correct at both endpoints.

| Transition | Both endpoints baseline-correct | Changed gate-consequence pattern | Rate | Runs with change |
|---|---:|---:|---:|---:|
| `pre_formation_to_formed` | 2153 | 1618 | 0.7515 | 13 |
| `formed_to_decline` | 632 | 524 | 0.8291 | 13 |
| `decline_to_recovered` | 660 | 515 | 0.7803 | 13 |

These are direct changes in intervention consequences, not cosine changes. They show that causal support relations reorganize during formation, decline, and recovery.

## Questions not identifiable from the current 11 conditions

- Single-component sufficiency is not identifiable without a triple gate that leaves one component active.
- The all-components-removed null is not identifiable without the four-component gate.
- A complete support lattice, unique minimal sufficient set, and interactions above pair order are not identifiable.
- A pair pattern compatible with backup or substitution cannot by itself exclude an unobserved higher-order interaction.
- Necessity and sufficiency are contextual to the executed target, query, receiving state, component versions, and gate site.

Therefore no extra forward is justified merely to complete a mathematical lattice. A future triple or four-component gate would be justified only for a specific unresolved sufficiency or higher-order causal question.

## Audit conclusion

The existing native interventions directly expose target-conditioned causal support organization: different concrete held-out instances retain the learned mod-23 relation under multiple different observed surviving configurations; single and pair gates produce target-boundary-specific necessity, pair-failure, and non-additive consequences; and these relations reorganize across formation, decline, and recovery.

**Claim boundary.** This is evidence for multiplicity and target-conditioned distributed causal organization in the executed 13-run nanoGPT family. It does not assert geometric convergence, a unique owner, a unique minimal support set, complete higher-order identification, or universal abstraction in all systems.

## Identity and integrity

All source GFGs reported `PASS`; every SQLite database, native logit tensor, and validation-target tensor was re-hashed before analysis. Query origins, target tensors, parameter versions, gate specifications, and native outcomes remain addressed by their source-GFG identities. Full per-run and per-target counts and hashes are in `FA_E02_EXISTING_EVIDENCE_AUDIT.json`.
