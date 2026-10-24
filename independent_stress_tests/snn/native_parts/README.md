# SNN native states (chunked binary)

The original experiment package contained 8 PyTorch .pt files, each 3,968,374 bytes. For every seed the replay .pt is **byte-identical** to the original .pt. This repository stores four distinct states as 16 raw binary parts (4 per seed) in Git, without duplicating replay bytes.

Restore all original formal and replay .pt files:

```bash
cd independent_stress_tests/snn
python restore_native_states.py
python snn_tlir_independent_checker.py --root formal --protocol snn_tlir_frozen_protocol.json
python snn_tlir_independent_checker.py --root replay --protocol snn_tlir_frozen_protocol.json
```

The restoration script concatenates parts in increasing index order, checks exact size and SHA-256 before accepting each original, and copies then rechecks replay. It does not deserialize PyTorch content. Only load .pt with torch.load if you trust this research artifact.

The original evidence integrity hashes and replay equality claims remain in formal/FORMAL_SUMMARY.json and SNN_TLIR_replay_comparison.json.
