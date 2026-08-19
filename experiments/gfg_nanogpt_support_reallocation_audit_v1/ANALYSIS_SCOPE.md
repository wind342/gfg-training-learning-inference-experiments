# Update-driven support reallocation audit

Status: `EXPLORATORY_POST_HOC`

This audit uses the existing finite-amplitude response corpus. For each section, alpha zero and alpha one are two branches from the same receiver state: no application of the recorded update and complete application of that update. No new training, GPU execution, response probe, or AI session is authorized.

For every target group with valid allocation vectors, the audit constructs

\[
S_0=(a_0,n_0,b_0,c_0,e_0,s_0,d_0),\qquad
S_1=(a_1,n_1,b_1,c_1,e_1,s_1,d_1),
\]

where `a` is component support allocation, `n` component necessity, `b` pair backup, `c` support concentration, `e` effective support, and `s,d` the single- and double-failure slack. The observed reallocation magnitude is

\[
R=\tfrac12\lVert a_1-a_0\rVert_1.
\]

The associated capability is the fraction of evaluation units in that target group predicted correctly before and after the update.

The audit reports exact before/after ledgers, rank correlations, primary-support switches, per-run consistency, and component-level relations. It is descriptive and causal within the frozen alpha-zero/alpha-one branch construction. It does not establish an advance predictor or a universal law.
