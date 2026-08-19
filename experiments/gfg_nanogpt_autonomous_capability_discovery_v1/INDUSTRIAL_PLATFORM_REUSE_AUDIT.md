# Industrial formal-session platform reuse audit

## Source

- source experiment:
  `experiments/gfg_tep_autonomous_discovery_paradigm_v1/`
- source branch:
  `experiment/gfg-tep-autonomous-discovery-paradigm-v1`
- original audited source commit:
  `7057270321839aa417e1b7b7cec6bd38c9732293`
- supplemental platform-repair source commit:
  `847e0eefef7842d87d2e5cbd9562b0239b53b123`

The source commit was inspected read-only. Its TEP mechanism, Fortran
execution, hidden probes, result tables and domain contracts were not copied.

## Reused platform mechanisms

- one isolated Multipass/Docker participant environment per formal instance;
- a fresh participant repository and Codex home;
- one `gpt-5.6-sol` process at `xhigh` effort;
- an externally controlled 300-second first-contact gate;
- a read-only evidence mount that is permission-locked before release;
- a 7,200-second formal-work budget starting only after release;
- web search disabled, ephemeral Codex execution and submission-only writes;
- session logs, output transfer, secret scanning and machine attestations;
- candidate admission followed by content-addressed sealing;
- platform failures kept distinct from scientific failures.
- participant Apps are disabled because they are outside the formal task;
- all uppercase and lowercase proxy variables are explicitly overridden
  after the container environment file so stale image defaults cannot shadow
  the audited participant proxy.

## nanoGPT-specific adaptations

- the participant evidence is the complete validated nanoGPT training GFG
  bundle, not TEP evidence;
- the same AI process must write a machine-validated
  `orientation_receipt.json` before the external runner releases the GFG;
- a local CONNECT proxy exposes only `chatgpt.com` and
  `auth.openai.com` to the isolated container and records destination-only
  audit rows; direct container egress remains unavailable;
- the nanoGPT executable mechanism, forecast and intervention contracts are
  unchanged by the platform reuse.

The proxy relay is an environment adapter for `gfg-lab-ubuntu-v3`: the VM can
reach the host proxy, while the isolated Docker bridge cannot reach it
directly. No scientific data pass through this relay.

After a Windows Default Switch change left the original instance's DHCP/DNS
identity stale, Multipass cloned it without deleting or modifying the source
instance. Formal execution uses the independently boot-verified
`gfg-lab-ubuntu-v5-stability` clone after a third Default Switch subnet
change. The participant image, Docker runtime,
Codex binary and scientific mounts are unchanged.

An end-to-end platform smoke then exposed two transport defects before a
scientific opportunity was consumed. The relay now preserves TLS bytes that
arrive in the same packet as the HTTP CONNECT header. Its host directory also
remains owned by the relay account, while the participant receives traverse
permission and read access only to its separate environment file. This keeps
the append-only destination audit writable by the relay but outside the
participant mounts. A real isolated-container Codex call through the repaired
allowlist returned the requested model response with 15 established tunnels
and zero denied destinations.

The first completed formal submission exposed a Multipass client defect when
the runner attempted to return the complete 10 KiB proxy audit through
`multipass exec`: the guest command had exited, but the Windows client did not
close its output stream. The candidate had already been transferred to the
host, while hidden-future generation had not started. Recovery therefore
preserves and seals those exact transferred bytes, reconstructs the session
attestation from the orientation receipt, model logs, submission manifest and
content-addressed raw proxy audit, and resumes only after submission.

The network audit also separates blocked attempts from successful egress. A
denied destination remains a recorded instruction-compliance signal, but it
does not invalidate the GFG-only evidence boundary because no tunnel was
established and no external bytes were delivered. Any established tunnel to a
destination outside the two-host allowlist remains a hard platform failure.

Before any hidden future was generated, candidate admission also exposed an
undisclosed field-name mismatch. The participant contracts require an
intervention direction and executable hook interface but do not prescribe
names for the predicted shift interval. The submitted candidate used the
unambiguous integer pair `transition_step_shift_low/high`; the validator had
only accepted the internal spelling
`predicted_transition_shift_low/high`. Admission and causal evaluation now
normalize either spelling to the same ordered pair. The submitted candidate
files remain byte-for-byte unchanged.

Candidate admission then exposed a second interface ambiguity before hidden
future generation. The published interface calls the input a GFG prefix and
documents `gfg_client.GFG` as the query interface, but the runtime had supplied
only a bare path string. The runtime now supplies a compatibility value that
remains a valid path string while delegating all documented GFG query methods
to the official read-only client. Path-oriented and object-oriented candidates
therefore receive the same frozen evidence without candidate rewriting.

Admission originally replayed the forecasting interface against the complete
10,000-step discovery run, even though the frozen scientific call occurs at
the pre-transition 500-step prediction cut. A correct mechanism may have no
future curve after the completed run. Admission now uses the registered
500-step prefix view, while the hidden run continues to use its independently
generated prediction-cut directory. Forecast rows accept both the platform's
short names (`step`, `accuracy`) and the candidate's explicitly declared
domain names (`optimizer_step`, `validation_accuracy`). Validator-generated
query records are written beside, rather than into, the transferred
submission so candidate bytes remain unchanged before sealing.

## Freeze boundary

The reused platform, its nanoGPT adapters, the orientation material and all
scientific contracts are included in the superseding protocol-freeze
manifest. The aborted pre-repair launch never completed orientation, released
target evidence, produced a candidate or consumed a scientific opportunity.
