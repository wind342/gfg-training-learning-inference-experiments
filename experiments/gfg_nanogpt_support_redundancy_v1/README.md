# nanoGPT Capability Support Redundancy GFG v1

This analysis-stage experiment performs real reversible component gates at every
materialized checkpoint of the 13 preserved nanoGPT training GFGs.  It measures
which of four residual branches support each opaque cyclic target group, which
pairs provide backup, how much single- and double-component failure slack
remains, and how the support allocation changes between adjacent checkpoints.

The source training GFGs preserve exact historical parameter, gradient, Adam,
input and evaluation objects.  The original CUDA driver/kernel environment was
not preserved and current raw logits differ from the historical stored logits
by a few float32 ulps while preserving predictions and capability.  This package
does not hide that boundary and does not mix old logits with new gate outputs.
It defines a new, content-bound GPU analysis execution from the exact historical
checkpoint states.  Two repeated ungated forwards must be byte-identical in the
current runtime before any support statistic is admitted.

For each checkpoint the package stores complete current baseline and gated
decision logits, predictions, row-order true-target margins, 23-group lower-tail
margins, component necessity, pair backup, single/double failure slack,
effective support, support turnover, and non-directional component optimizer
loads.  Large arrays are content-addressed `.npy` outcomes.  Every derived value
is formed by an explicit occurrence and atomic `f=(u,tau,omega_bar,z;rho)`
bindings with GeneratedOrigin paths to the exact source training objects.

No capability phase label, suffix fact, future performance, direction
hypothesis or newly fitted stability law enters the measurement.

Query one bundle:

```text
python participant_query.py BUNDLE summary
python participant_query.py BUNDLE step 5000
python participant_query.py BUNDLE series component_target_group_necessity
python participant_query.py BUNDLE series support_turnover
```
