# Evidence packaging and scientific scope

Archive identity recorded from the original conversation artifacts:

| Study | Original archive | SHA-256 | Total files |
|---|---|---|---:|
| GNN | GNN_TLIR_反证实验完整证据包.zip | 1c607ca8d775b430c01fe38c60ecb5a5228e80d5de6bed22036259a555202716 | 11 |
| SNN | TLIR_硬脉冲网络_独立反证证据包.zip | 3121edcd2308e57a7c06e66540512062afa611472e96c249e9e34d071ee5c6da | 39 |

GNN has 11 UTF-8 text files. SNN has 31 UTF-8 text files and eight .pt native-state files: four unique per-seed original files, plus four byte-identical replay copies. For per-seed hashes consult snn/formal/FORMAL_SUMMARY.json and snn/SNN_TLIR_replay_comparison.json. Replicated binary data does not create independent new scientific states.

The experiments used PyTorch-native custom measurements rather than original GFG Core v3 execution and full four-instrument implementation. Evidence hashes check identity/integrity, not a universal scientific claim. They do not provide an independent laboratory reproduction nor test RL feedback closure.

The original scripts may depend on /mnt/data paths; this is preserved intentionally rather than silently rewriting frozen source. Full original ZIP archives were generated inside the ChatGPT working session; this GitHub branch preserves the text source/record files and relevant distinct native-state files when supplied.
