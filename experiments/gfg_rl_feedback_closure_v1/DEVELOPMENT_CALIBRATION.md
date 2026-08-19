# Development calibration record

The first immutable development run is stored at
`E:\gfg-evidence\rl-feedback-closure-v1\development-v2` under contract hash
`878f5fe537f48814725fec6cd7094d79c3329445c210ae97b44dc248300eb4bfb`.

It passed the phase-A formation gate and showed a large directional A-over-B/C
advantage, but condition A reached only 0.25, 0.4375 and 0.375 final chain
accuracy.  The actual curves showed a saturated old policy rather than an
unlearnable task: all phase-A seals were 1.0 while phase-B adaptation plateaued.

Development-only probes therefore varied the phase-A seal budget and entropy
coefficient without opening any formal seed.  Earlier sealing alone did not
remove the plateau.  Entropy coefficient 1.0 retained phase-A chain accuracy
1.0 and yielded phase-B condition-A chain accuracy 1.0 in all three development
seeds after 300 reversal updates.  The formal contract consequently fixes:

- 160 phase-A updates;
- 300 phase-B updates;
- entropy coefficient 1.0;
- behavior exploration epsilon 0.20 with the declared policy-gradient
  correction.

No formal seed or formal outcome was read during this calibration.  The failed
first development run is retained and is not reported as a formal result.
