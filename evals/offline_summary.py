"""Offline recovery of stage-11 sweep evidence when LangSmith trace
ingestion fails mid-sweep.

2026-08-09: eight stage-11 cells exited 0 but the workspace's monthly trace
spend cap was reached partway through, so LangSmith recorded most rows as
"no run for this example" (`insights.md`). The agents still ran and
`telemetry.py` still wrote `output/eval-runs/*/logs/*.jsonl` locally --
that write never depended on the LangSmith connection succeeding.

Only two things are honestly recoverable from those local files:
`revision_converged` (`-verdicts.jsonl` carries exactly what that evaluator
reads, so this module calls the real, tested `evaluators.revision_converged`
rather than re-implementing its logic) and a rough tool-call trajectory
shape (`-tools.jsonl`: which tool, which agent, ok/error). Cost, tokens,
latency and `verdict_is_justified` are not recoverable here -- `tools.jsonl`
never carried token/cost data (LangSmith computed those from the trace
payloads that never arrived), and `verdict_is_justified` needs the Critic's
full rendered text, which telemetry only ever recorded the *length* of.
This is supplementary evidence for the write-up, not a replacement for a
sweep with ingestion actually working.

Attribution works by time window because each stage-11 cell that produced
this data ran as its own process (`evals/experiments.py --only PREFIX`),
sequentially, one at a time -- so a run directory's first telemetry
timestamp falls inside exactly one cell's wall-clock window, with no
overlap to resolve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.evaluators import revision_converged


@dataclass(frozen=True)
class CellWindow:
    """One matrix cell's wall-clock run window, for timestamp attribution.

    Parameters
    ----------
    name : str
        The cell's `experiment_prefix`.
    start, end : str
        ISO-8601 timestamps (any format that sorts lexicographically in
        timestamp order, e.g. `datetime.isoformat()`). `start` inclusive,
        `end` exclusive.
    max_revisions : int
        The `max_revisions` this cell ran with -- forwarded to
        `evaluators.revision_converged`, which needs it to tell a
        deliberate cap-out from a run that simply never converged.
    """

    name: str
    start: str
    end: str
    max_revisions: int = 2


@dataclass
class CellSummary:
    """What survived locally for one matrix cell.

    `revision_converged_rate` is a running mean over every run whose
    `revision_converged` score was not `None` -- backed by two raw counters
    rather than a stored average, so folding in one more run needs no
    knowledge of how many were folded in before.
    """

    n_runs: int = 0
    tool_calls_by_agent: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    _converged_total: float = 0.0
    _converged_count: int = 0
    _first_round_approve_total: float = 0.0
    _first_round_approve_count: int = 0
    _final_round_total: float = 0.0
    _final_round_count: int = 0

    @property
    def revision_converged_rate(self) -> float | None:
        """Whether the run ended `APPROVE` or a deliberate cap-out.

        Delegates to `evaluators.revision_converged` -- the plan's chosen
        metric for stage 11 experiment 2.
        """
        if self._converged_count == 0:
            return None
        return self._converged_total / self._converged_count

    @property
    def first_round_approve_rate(self) -> float | None:
        """Fraction of runs the Critic approved without any revision.

        Reads only the *first* verdict event, unlike `revision_converged`
        (which reads the last) -- a run that revised once and then
        converged is `revision_converged=True` but
        `first_round_approve=False`; the two answer different questions.
        Stage 11 experiment 4's chosen metric, alongside
        `verdict_is_justified` (not recoverable offline).
        """
        if self._first_round_approve_count == 0:
            return None
        return self._first_round_approve_total / self._first_round_approve_count

    @property
    def mean_final_round(self) -> float | None:
        """Average round number of the last verdict per run.

        The plan's "rounds actually taken" half of experiment 2's metric,
        alongside `revision_converged_rate`.
        """
        if self._final_round_count == 0:
            return None
        return self._final_round_total / self._final_round_count


def assign_window(timestamp: str, windows: list[CellWindow]) -> str | None:
    """Return the name of the window containing `timestamp`, or `None`.

    Windows are assumed non-overlapping; the first match wins.
    """
    for window in windows:
        if window.start <= timestamp < window.end:
            return window.name
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL telemetry stream, tolerating a corrupted line.

    The same interleaved-write race `evaluators._read_jsonl` already
    tolerates (concurrent `ToolNode` tool calls each write through their
    own `TelemetryHandler.on_tool_end`) hits this reader too -- confirmed
    against the real 2026-08-09 sweep data, not assumed. One bad line must
    not lose every other event in the file.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _first_timestamp(run_dir: Path) -> str | None:
    logs_dir = run_dir / "logs"
    for name in ("verdicts", "tools"):
        events = _read_jsonl(logs_dir / f"{run_dir.name}-{name}.jsonl")
        if events and "ts" in events[0]:
            return str(events[0]["ts"])
    return None


def summarize(output_root: Path, windows: list[CellWindow]) -> dict[str, CellSummary]:
    """Bucket every run directory under `output_root` into its cell window.

    Parameters
    ----------
    output_root : Path
        A directory of `eval-<uuid>` run directories, each holding
        `logs/<id>-verdicts.jsonl` and `logs/<id>-tools.jsonl`
        (`evals/run_eval.py`'s per-example layout).
    windows : list of CellWindow
        The matrix cells to attribute runs to. A run whose first timestamp
        falls outside every window is silently dropped -- it belongs to a
        different sweep, not this one.

    Returns
    -------
    dict
        `cell name -> CellSummary`, one entry per window, present even when
        a cell matched zero runs.
    """
    results = {window.name: CellSummary() for window in windows}
    by_window = {window.name: window for window in windows}

    if not output_root.exists():
        return results

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ts = _first_timestamp(run_dir)
        if ts is None:
            continue
        cell_name = assign_window(ts, windows)
        if cell_name is None:
            continue

        summary = results[cell_name]
        summary.n_runs += 1

        verdicts_log = run_dir / "logs" / f"{run_dir.name}-verdicts.jsonl"
        outcome = revision_converged(
            outputs={
                "verdicts_log": str(verdicts_log),
                "max_revisions": by_window[cell_name].max_revisions,
            }
        )
        score = outcome["score"]
        if score is not None:
            summary._converged_total += 1.0 if score else 0.0
            summary._converged_count += 1

        verdict_events = _read_jsonl(verdicts_log)
        if verdict_events:
            summary._first_round_approve_total += (
                1.0 if verdict_events[0].get("verdict") == "APPROVE" else 0.0
            )
            summary._first_round_approve_count += 1
            summary._final_round_total += float(verdict_events[-1].get("round", 0))
            summary._final_round_count += 1

        for event in _read_jsonl(run_dir / "logs" / f"{run_dir.name}-tools.jsonl"):
            agent = str(event.get("agent", "unknown"))
            summary.tool_calls_by_agent[agent] = (
                summary.tool_calls_by_agent.get(agent, 0) + 1
            )
            if not event.get("ok", True):
                summary.tool_errors += 1

    return results
