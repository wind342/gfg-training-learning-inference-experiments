# Discovery report: burst-driven cyclic-equivariance formation

## Result

The selected mechanism is a finite-state optimizer/circuit mechanism, not an
accuracy-curve label. Full-batch AdamW training enters separated high-gradient
episodes. Each episode is clipped and then consumed by concrete optimizer
occurrences. Across episodes, the token embedding becomes increasingly
translation-equivariant in the cyclic coordinate system implied by the
observed training facts. The model changes state from memorization to rule
generalization after the forecast number of discrete rewrite episodes.

At the prefix cut, the detected non-initialization burst peaks are optimizer
steps 189, 433, 729, and 1041. Their spacings are 244, 296, and 312 steps. A
bounded finite-difference extrapolation predicts the next centers near 1387
and 1767; binding the latter to the frozen 100-step evaluation grid predicts
transition step 1800. The sealed high-precision interval is 1700--1900, and
the primary interval is 1600--2000. Both intervals are computed from an unseen
prefix by `mechanism.py`; they are not selected by run identity.

## Exact task-relative order parameter

The materialized batch has 317 atomic observed equations. For a row with
input tokens `(x, operator, y)` and last target `z`, modular row reduction of
the equations `coordinate(z) = coordinate(x) + coordinate(y)` recovered a
bijective cyclic coordinate assignment for all 23 non-operator tokens. The
candidate uses that assignment transiently and never constructs or serializes
a completed answer table.

For each materialized evaluation-boundary `transformer.wte.weight`, I centered
the 23 task-token rows, formed their Gram matrix, and measured the fraction of
off-diagonal Gram variance explained by cyclic displacement. This quantity is
invariant to a rotation or nonzero rescaling of the recovered cyclic
coordinate. Selected observations are:

| optimizer step | validation accuracy | cyclic equivariance |
| ---: | ---: | ---: |
| 100 | 0.037736 | 0.266136 |
| 500 | 0.084906 | 0.523572 |
| 800 | 0.169811 | 0.618131 |
| 1100 | 0.311321 | 0.728069 |
| 1400 | 0.570755 | 0.825162 |
| 1500 | 0.783019 | 0.854683 |
| 1800 | 0.971698 | 0.886437 |

The step-1100 parameter is
`obj_af2e4d72edd8b1efff4c8ed065a01b0c23784a1939e13dd3304f5283b1a80cb9`
and the bound evaluation is occurrence
`occ_670715dcaa9f7f1ead5e99e4a6aba7b3438d96eebf5f1cedc53c12b6f38a12b9`.
At step 1800 the corresponding identities are parameter
`obj_6ae603da019c3ca87751a60127a54886c7485f9fd42ad068d64c4b619bb59390`
and evaluation occurrence
`occ_e05657e8df7aead4233d526796ab6cc2b585543823ce0ba42d08c5098ed0c471`.
These are exact object and occurrence identities, not joins by step or value.

## Forward GFG chain at two rewrite episodes

At optimizer step 1041, autograd occurrence
`occ_c7413f77567d3e80c2a49dd45bef2679987ef49ededbb5e6a496b910201c75a8`
formed the WTE gradient
`obj_1a0ebfceaf5d2226051e1723e33e926bc3124e30c86a786c48a7c1a0b8978d69`
in fact block
`factblock_58b3ad501c703dee361eb5576916582f6a6a8502f3de5a3bbe85ca8042985733`.
Clip occurrence
`occ_b9a33354cfc43eb956b208e3cd5e27bf2285ab8ee6c01f63b63d40ef245bb9be`
recorded total norm 47.946800 and formed the distinct clipped-gradient object
`obj_e7210890e937f0c5d5bd1e285f0aebad1d772e3be6fc4641c077f191828e8e7d`.
Optimizer occurrence
`occ_eb8fdb65c4b39d33a719791efe73aa536f56580b98e14d7046d776adc4e31138`
then formed WTE version 1042,
`obj_95e003300876d7ae186b95cad34483fa3f85aac05ed60c1648082f3cc0e96704`.
Primitive `GeneratedOrigin` edge
`edge_dae06b34939e754fd6181b1e5fe6e48b9cf166edb7c48d50652dc73d8ccdf952`
connects the exact prior WTE version to that exact result.

The decisive later episode is independently identified. At step 1714,
autograd occurrence
`occ_bfbbcf64968b7cbb104c764b5b16302b4aaf1bc1e0546ef1da109abba3236c4e`
formed gradient
`obj_9d7b5b3993a565eb04f900af1031bc3d13ecdc4d70890d5479f8c6e1e53e2151`.
Clip occurrence
`occ_580453a6a1408d8e2dd9a0b0ef8f02091cb0652aa26ac8ece6d0d92b2ffe45c1`
recorded total norm 69.200523 and formed clipped gradient
`obj_b24927f26fe5d31a6f86d62c4170c88fe70b4212c04719a62809beb6d6811a1b`.
Optimizer occurrence
`occ_ec0c7642a72f4a4763d24141891f64a5e013b55f6e5486cd8b3c8f05889e4826`
formed WTE version 1715,
`obj_bcc0179f54d6eb6a91e605bc5d98e57e005747b93e0f278263bb08676067b590`,
with primitive parameter-version edge
`edge_ae618a2ecb33a520ab8321b9ec42f2fcd3d6f48f9250cabd9fbcefa8e47b4fe2`.

The transition evaluation itself is fact block
`factblock_628c75a87ceac61b26b49592ab8877c51cb5cccce85f5e37e964a99db8af78d5`.
Its evaluation occurrence realizes separate facts from the exact evaluated
parameter sources to train predictions, train logits, validation predictions,
validation logits, and metric object
`obj_8485a030b6b528a645dd57454f3f9aedc9b5113754e873eaa594d8309c1ea1ea`.
I did not split those facts into source/result sets or recombine them.

The frozen transition definition is satisfied at 1800: steps 1800, 1900,
and 2000 have train accuracy 1.0 and validation accuracies 0.971698,
0.981132, and 0.985849. An earlier evaluation at step 1000 has validation
accuracy 0.183962. Thus 1800 is the first point of the first qualifying
three-point window.

## Falsification attempts

### Selected mechanism: burst-driven cyclic equivariance

I tried to falsify the order parameter by destroying the inferred task
coordinate while retaining the same WTE values. Across 100 deterministic
random token permutations, the permutation-null 95th percentile was 0.085155
at step 1100 versus the observed 0.728069, and 0.096142 at step 1800 versus
0.886437. The effect is therefore not a generic consequence of Gram scale or
its diagonal.

I also tested whether cyclic equivariance alone was sufficient. It was not:
step 3900 retained score 0.827069 while validation accuracy was 0.221698.
This falsifies a one-scalar permanent-capability account. The selected finite
state therefore requires both representation order and optimizer-burst/readout
phase, and it explicitly forecasts a later `REGULARIZATION_LIMIT_CYCLE` state.

### Rejected alternative 1: training-loss or memorization threshold

This alternative predicts transition once training has been fitted. It fails
sharply. At step 100, evaluation occurrence
`occ_0c1b875c44c35453540a0fae81dc1a585bdeb5ca25f12a089c63dc12366f97af`
already has train accuracy 1.0 and loss 0.011849, but validation accuracy is
0.037736. At step 1000, train accuracy remains 1.0 and loss has fallen to
0.000231 while validation accuracy is only 0.183962. Memorization is a
necessary prefix condition, not the executable state change.

### Rejected alternative 2: one gradient-norm or clipping threshold

If crossing a gradient norm threshold directly caused generalization, the
large earlier bursts would already qualify. The pre-cut burst peaks include
29.363251 at step 189, 35.220222 at 433, 52.733410 at 729, and 47.946800 at
1041, while the next-grid validation accuracies remain 0.033019, 0.084906,
0.169811, and 0.311321 respectively. Conversely, the transition-grid clip
record at step 1800 has norm only 0.002262. A burst is a rewrite event, but
cumulative task-relative representational state determines its effect.

### Rejected alternative 3: a fixed parameter-norm threshold

The WTE centered RMS falls under weight decay but is not an identifying state:
it is 0.0891 at step 1100, 0.0768 at 1200, 0.0723 at transition step 1800,
and 0.0485 at step 3900, where validation has collapsed to 0.221698. Equal or
smaller norms do not establish equal formation, and no fixed norm boundary
separates all rule-level and failed states.

## Forecast and intervention

The forecast curve is generated from the burst-indexed states. Before the next
rewrite it holds the observed prefix accuracy; after the first future rewrite
it enters consolidation; after the final predicted rewrite it enters
rule-level accuracy. A late lower mean is emitted when the regularized
oscillator enters its observed limit-cycle regime. The code emits every future
100-step grid point and the corresponding declared state.

The intervention is `DELAY`. At `before_optimizer_step`, it sets the learning
rate of every existing optimizer parameter group to zero for 1300 optimizer
calls and then restores each captured learning rate exactly. This is solely an
allowed optimizer-group-hyperparameter mutation. It reads no validation or
future object and injects no parameter values. Optimizer occurrences and their
generation facts remain distinct during the hold even where their parameter
outcomes have equal numerical contents.

The predicted shift interval is +600 to +1800 optimizer steps. During the
hold, the parameter trajectory cannot execute the predicted symmetry-rewrite
updates, so the mechanism remains in `SYMMETRY_NUCLEATION`; Adam moments may
still evolve, which is why the upper endpoint is deliberately wider than the
1300-step nominal hold. The direction and minimum shift are sealed before the
intervened future.

## Query boundary

All scientific inputs came from the released participant-safe GFG. Queries
were read-only. Materialized tensor bytes were content-hash checked before the
derived coordinate and Gram calculations. Occurrence identity,
`realizes_fact`/fact-block incidence, actual source roles, `GeneratedOrigin`,
and parameter-version identity supplied temporal authority; numeric similarity
and `program_order` alone were never used as dependency evidence. No network,
external literature, hidden material, run/date lookup, or additional AI call
was used.
