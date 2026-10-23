# Independent TLI/TLIR falsification stress tests (2026-10-24)

Independent ChatGPT-assisted experiments on two new neural systems. **These are not native GFG Core v3 reproductions and do not execute the original four GFG instruments.** Some intervention methods are inspired by those experiments. Neither study tests the reinforcement/feedback closure part of TLIR.

## Experiments

- [GNN, four-layer residual message-passing, RMSprop](gnn/GNN_TLIR_独立反证实验报告.md): 3 formal seeds; one frozen finite-amplitude nonlinear-frequency gate **FAILED** (seed 20261024: 4/256 = 1.56% versus a 10% threshold), while both other seeds passed this gate. This formal negative result is retained.
- [Hard-spiking recurrent SNN, Adagrad](snn/TLIR_脉冲网络独立反证实验报告.md): 4 formal seeds, all 12 operational checks passed per seed; single-branch low-response cases and surrogate-derivative mismatch are retained.

## Source integrity and reproduction limits

All original source, frozen protocol, text/numerical results, replay logs, checkers and reports are included without reclassifying them as primary research evidence. The SNN native PyTorch .pt states require a trusted source (do not load untrusted pickle) and hashes are documented by the formal manifest and replay comparison.

Some scripts embed absolute /mnt/data source paths, which were **preserved to retain the original frozen hashes**. A new-directory reproduction may require a matching path layout or a clearly disclosed working-copy path edit; it then is not a byte-identical original-source replay.

Original publication tag and core experiments/ remain untouched. See [evidence scope](EVIDENCE_SCOPE.md).
