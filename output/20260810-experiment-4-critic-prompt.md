# Experiment 4 — Critic prompt `c1` vs `c2`

**Question.** `c1` instructs the Critic: *"Return verdict APPROVE only when
all three dimensions hold; otherwise return REVISE."* That is a necessary
condition, not a biconditional — a run can satisfy it while still returning
`REVISE` with all three dimensions `True`, or `APPROVE` with a non-empty
`gaps` list, and both were observed for real across independent runs in
stages 5–8. `c2` closes the gap with one added clause: *"return APPROVE
exactly when [...] AND gaps and revision_requests are both empty; [...]
never a mismatched APPROVE [...] never a REVISE alongside three True
booleans and nothing left to fix."* Does the explicit biconditional actually
change the rate at which the verdict is internally consistent?

**Metric.** `verdict_is_justified` — a real 0.78 → 0.64/0.50 spread was
measured across two identical sweeps at n=18 (~14 percentage points), so this
experiment runs **three independent 18-row sweeps per arm**, not one; a
single sweep cannot tell the prompts apart at this dataset size. Runs on the
**graph path only**: `supervisor.py`'s `plan`/`research`/`critique`
wrappers call `load_settings()` internally, so a `critic_prompt_version`
override never reaches them on the supervisor path, and both arms would
silently measure the same configuration twice.

**Data source.** Six real LangSmith experiments, 18/18 rows each, zero
errors — this sweep ran clean, after the trace-ingestion issue that affected
an earlier attempt was resolved (workspace trace spend cap raised) and after
fixing a concurrency defect in the retriever's cross-encoder cache
(`retriever.py`, unrelated to this experiment's own result but a
precondition for a sweep completing at all). `first_round_approve_rate` is
supplementary, recovered offline from the same local telemetry
Experiment 2 uses.

## Results (18 rows per cell)

| Arm | run | verdict_is_justified | first-round APPROVE | mean final round | tokens | cost |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 1 | 0.58 | 0.72 | 0.44 | 858,410 | $0.260 |
| c1 | 2 | 0.69 | 0.67 | 0.61 | 823,243 | $0.255 |
| c1 | 3 | 0.72 | 0.44 | 0.89 | 1,051,814 | $0.322 |
| **c1 mean** | | **0.66** | **0.61** | **0.65** | **911,156** | **$0.279** |
| c2 | 1 | 0.92 | 0.50 | 0.67 | 844,136 | $0.253 |
| c2 | 2 | 0.97 | 0.56 | 0.72 | 919,146 | $0.271 |
| c2 | 3 | 0.97 | 0.56 | 0.61 | 799,195 | $0.254 |
| **c2 mean** | | **0.95** | **0.54** | **0.67** | **854,159** | **$0.259** |

`revision_converged` and `critic_verified` are both 1.00 on every one of the
six cells — saturated as expected, no signal. `first-round APPROVE` and
`mean final round` are recovered offline from the same local telemetry
Experiment 2 uses, attributed to these same six cells by wall-clock window;
`verdict_is_justified`, tokens and cost are LangSmith's own live numbers
for this sweep.

## Findings

**`c2` wins decisively, not marginally.** The three-sweep ranges do not
overlap: `c1`'s best sweep (0.72) is still below `c2`'s worst (0.92). The
gap between arms (0.20–0.39) is larger than the ~0.14 spread already
measured for a single sweep's own noise at this dataset size, so this reads
as a real effect, not sampling variance dressed up as one.

**No cost penalty — `c2` is marginally cheaper.** Mean tokens: 911k (`c1`)
vs 854k (`c2`), about 6% fewer; mean cost: $0.279 vs $0.259, about 7% less.
The explicit consistency clause did not make the Critic do more work to
reach a verdict.

**`c2` approves on the first pass somewhat less often — a plausible
trade-off, not a hidden cost.** `first_round_approve_rate` drops from a
mean of 0.61 (`c1`) to 0.54 (`c2`); the three `c2` sweeps also cluster more
tightly (0.50–0.56) than `c1`'s (0.44–0.72). A Critic that can no longer
rubber-stamp an `APPROVE` alongside an unresolved `gaps` entry has room to
send a few more runs back for one extra round instead — a plausible
mechanism behind the consistency gain above, though this experiment does
not isolate it as the sole cause.

## Conclusion

`c2` measurably closes the gap it was written for, at no cost penalty.
**`config.py`'s `critic_prompt_version` default changes from `"c1"` to
`"c2"`**, with this measurement cited inline in the field's comment — the
same convention the project already uses for other measured defaults.
