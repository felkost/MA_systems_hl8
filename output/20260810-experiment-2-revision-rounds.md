# Experiment 2 — does `max_revisions` change what the revision loop achieves?

**Question.** Does raising the Critic's revision budget (`max_revisions`)
change whether a run actually reaches a resolved state, or does the extra
budget go unused? The assignment ships `max_revisions=2`; this experiment
measures 1, 2 and 3 on both orchestration paths to see whether that default
is well-chosen.

**Metric.** `revision_converged` (did the run end `APPROVE`, or hit a
deliberate cap-out at `round >= max_revisions`?) plus the mean final round
number — the two components the project's evaluation plan names for this
experiment, chosen because four of the six end-to-end evaluators are
saturated at 1.00 on every configuration and cannot discriminate anything
(see the stage 9/R2 findings).

**Data source.** All six cells below are recovered from local telemetry
(`output/eval-runs/*/logs/*-verdicts.jsonl`), not from LangSmith's own
aggregates. The LangSmith experiments that produced this data were run
against a dataset that was later deleted and recreated mid-session, which
orphaned their trace records without touching the local files `telemetry.py`
had already written — `evals/offline_summary.py` re-attributes each local
run directory to its cell by wall-clock window (each cell ran as its own
process, sequentially, so windows never overlap). Cost, tokens and latency
are **not** recoverable this way (LangSmith computes those from the trace
payloads that were lost, and `telemetry.py` never records them at all), so
this experiment reports convergence only — the path-level cost comparison
is Experiment 1's, unaffected by this incident.

## Results (18 rows per cell, n=18)

| Path | max_revisions | revision_converged | first-round APPROVE | mean final round |
| --- | --- | --- | --- | --- |
| supervisor | 1 | 0.94 | 0.61 | 0.50 |
| supervisor | 2 | 0.78 | 0.67 | 0.39 |
| supervisor | 3 | 0.72 | 0.56 | 0.56 |
| graph | 1 | 1.00 | 0.50 | 0.50 |
| graph | 2 | 1.00 | 0.67 | 0.44 |
| graph | 3 | 1.00 | 0.56 | 0.83 |

## Findings

**The graph path converges every time, regardless of the budget.** This is
structural, not a property of the budget: `orchestrator.py`'s
`_route_after_critique` sends `APPROVE` straight to `write` and forces a
cap-out otherwise, so every run either approves or deliberately exhausts its
rounds. `revision_converged` cannot be anything but 1.00 here by
construction — a ceiling effect, not a finding about `max_revisions` itself.

**On the supervisor path, a bigger budget does not buy more resolved runs —
if anything, the opposite.** `revision_converged` falls from 0.94 at
`max_revisions=1` to 0.78 at 2 and 0.72 at 3. A `False` score here means the
run ended on `REVISE` with `round < max_revisions` — the model stopped
short of both an approval and the cap. The likely mechanism, not confirmed
by this experiment: `ToolCallLimitMiddleware`'s own call budget
(`run_limit=1 + max_revisions` on `research`/`critique`) is a *count of tool
calls*, not the same quantity as the round counter `revision_converged`
reads, and the model is free to spend that budget on a discretionary extra
call (stage 7 already documented the Supervisor choosing an extra `research`
call after an `APPROVE`) — a larger nominal budget gives the model more
room to spend it in a way that leaves the round counter short of the cap
without a final verdict resolving it. This would need trace-level
inspection of the unconverged rows to confirm; it is offered here as the
most plausible explanation, not a proven cause.

**Neither path shows a clean trend in rounds actually taken.** Mean final
round moves non-monotonically with the budget on both paths (0.50 → 0.39 →
0.56 supervisor; 0.50 → 0.44 → 0.83 graph) — consistent with the dataset's
18 questions varying in how much revision they call for, rather than with
raising the cap itself changing typical behaviour.

## Conclusion

`max_revisions=2` — the assignment's own default — sits in the middle of
both effects measured here: it is not the supervisor path's best-converging
setting (that is 1), nor does the graph path care at all. There is no
evidence in this data that raising the cap to 3 improves anything on either
path, and on the supervisor path it measurably costs convergence. Per the
project's standing decision, **`max_revisions` ships at 2 regardless** — the
assignment specifies it, and this measurement is recorded as a note against
that default rather than a reason to change it. The finding worth carrying
forward is the asymmetry itself: the graph path's deterministic routing is
what actually guarantees a resolved run, not the size of the budget it is
given.
