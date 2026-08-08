"""Turn the JSONL telemetry `telemetry.py` writes into CSV, PNG and Markdown.

Run as a separate command: `python reports.py`. Reads every
`output/logs/*-tools.jsonl`, `output/logs/*-retrieval.jsonl` and
`output/logs/*-verdicts.jsonl` file across all sessions, so nothing here
needs a live agent process. Works offline: LangSmith (layer 1 of the plan's
three observability layers) is a future augmentation, never a requirement,
so this module does not import it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no GUI backend: this runs headless and in CI

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

import paths  # noqa: E402
from config import Settings, load_settings  # noqa: E402

CHART_TOOL_CALLS_BY_AGENT = "tool-calls-by-agent.png"
CHART_TOOL_LATENCY_BY_AGENT = "tool-latency-by-agent.png"
CHART_TOOL_ERRORS = "tool-errors.png"
CHART_VERDICT_DISTRIBUTION = "verdict-distribution.png"
CHART_REVISION_ROUNDS = "revision-rounds.png"
CHART_ARM_CONTRIBUTION = "arm-contribution.png"
CHART_RERANK_EFFECT = "rerank-effect.png"

ARM_LABELS = ("semantic", "lexical", "both", "unknown")


@dataclass(frozen=True)
class ReportPaths:
    """Where one `build_report` run wrote its output."""

    markdown: Path
    csv: Path
    charts_dir: Path


# -- reading ------------------------------------------------------------


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL telemetry file.

    Returns an empty list for a file that does not exist yet, so a report
    can be built before any session has run.
    """
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_all_tool_events(settings: Settings) -> list[dict[str, Any]]:
    """Merge every session's tool-call stream."""
    return _read_all(_log_dir(settings), "-tools.jsonl")


def read_all_retrieval_events(settings: Settings) -> list[dict[str, Any]]:
    """Merge every session's retrieval stream."""
    return _read_all(_log_dir(settings), "-retrieval.jsonl")


def read_all_verdict_events(settings: Settings) -> list[dict[str, Any]]:
    """Merge every session's Critic-verdict stream."""
    return _read_all(_log_dir(settings), "-verdicts.jsonl")


def _log_dir(settings: Settings) -> Path:
    return paths.resolve(settings.output_dir) / "logs"


def _read_all(log_dir: Path, suffix: str) -> list[dict[str, Any]]:
    if not log_dir.is_dir():
        return []
    events: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob(f"*{suffix}")):
        events.extend(read_events(path))
    return events


# -- aggregation, per-agent -------------------------------------------------


def tool_call_counts_by_agent(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """How many times each agent called each tool."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        counts[event["agent"]][event["tool"]] += 1
    return {agent: dict(tools) for agent, tools in counts.items()}


def tool_latency_by_agent(
    events: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Median and p95 seconds per agent, across every tool it called."""
    by_agent: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_agent[event["agent"]].append(event["seconds"])
    return {agent: _median_and_p95(seconds) for agent, seconds in by_agent.items()}


def _median_and_p95(seconds: list[float]) -> tuple[float, float]:
    ordered = sorted(seconds)
    return _percentile(ordered, 50), _percentile(ordered, 95)


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def tool_error_rates(events: list[dict[str, Any]]) -> dict[str, float]:
    """Share of failed calls per tool -- how reliable `read_url` really is."""
    by_tool: dict[str, list[bool]] = defaultdict(list)
    for event in events:
        by_tool[event["tool"]].append(bool(event["ok"]))
    return {tool: 1 - (sum(oks) / len(oks)) for tool, oks in by_tool.items()}


# -- aggregation, verdicts and revision rounds -------------------------------


def verdict_distribution(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """APPROVE/REVISE counts per coordination path.

    Neither orchestration path exposes a verdict as a tool call, so this
    reads `telemetry.log_verdict`'s own stream, not the tool-call one --
    the coordination-path comparison stage 11 needs is exactly what this
    grouping is for.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        counts[event["path"]][event["verdict"]] += 1
    return {path: dict(verdicts) for path, verdicts in counts.items()}


def revision_rounds_per_session(events: list[dict[str, Any]]) -> dict[str, int]:
    """The highest revision round each session reached.

    0 means approved (or rejected) on the first pass, with no revision at
    all -- the same round numbering `telemetry.log_verdict` callers use.
    """
    rounds: dict[str, int] = {}
    for event in events:
        session = event["session"]
        rounds[session] = max(rounds.get(session, 0), event["round"])
    return rounds


# -- aggregation, retrieval (unchanged by agent count) -----------------------


def arm_contribution(events: list[dict[str, Any]]) -> dict[str, int]:
    """Of the documents that reached the final answer, which arm found them.

    "both" means the ensemble already agreed before the reranker ran;
    "unknown" would mean a document survived reranking without appearing in
    either arm's own result, which should not happen and is kept visible
    rather than silently dropped if it ever does.
    """
    counts = {label: 0 for label in ARM_LABELS}
    for event in events:
        semantic_ids = set(event.get("semantic_ids", []))
        bm25_ids = set(event.get("bm25_ids", []))
        for item in event.get("reranked", []):
            in_semantic = item["id"] in semantic_ids
            in_bm25 = item["id"] in bm25_ids
            if in_semantic and in_bm25:
                counts["both"] += 1
            elif in_semantic:
                counts["semantic"] += 1
            elif in_bm25:
                counts["lexical"] += 1
            else:
                counts["unknown"] += 1
    return counts


def rerank_movement(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """`(rank_before, rank_after)` for every document the reranker kept."""
    return [
        (item["rank_before"], item["rank_after"])
        for event in events
        for item in event.get("reranked", [])
        if item.get("rank_before") is not None
    ]


# -- charts -----------------------------------------------------------------


def plot_tool_calls_by_agent(counts: dict[str, dict[str, int]], path: Path) -> None:
    """Stacked bar: which agent drives tool usage, and with which tool."""
    fig, ax = plt.subplots()
    agents = sorted(counts)
    tools = sorted({tool for tool_counts in counts.values() for tool in tool_counts})
    bottom = [0.0] * len(agents)
    for tool in tools:
        values = [counts[agent].get(tool, 0) for agent in agents]
        ax.bar(agents, values, bottom=bottom, label=tool)
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    ax.set_ylabel("calls")
    ax.set_title("Tool calls by agent")
    if tools:
        ax.legend()
    _save(fig, path)


def plot_tool_latency_by_agent(
    stats: dict[str, tuple[float, float]], path: Path
) -> None:
    """Grouped bars: median and p95 seconds per agent."""
    fig, ax = plt.subplots()
    agents = sorted(stats)
    positions = range(len(agents))
    width = 0.35
    ax.bar(
        [p - width / 2 for p in positions],
        [stats[agent][0] for agent in agents],
        width,
        label="median",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [stats[agent][1] for agent in agents],
        width,
        label="p95",
    )
    ax.set_xticks(list(positions), agents)
    ax.set_ylabel("seconds")
    ax.set_title("Tool latency by agent")
    ax.legend()
    _save(fig, path)


def plot_tool_error_rates(rates: dict[str, float], path: Path) -> None:
    """Bar chart: how unreliable each tool is, `read_url` vs the rest."""
    fig, ax = plt.subplots()
    tools = sorted(rates)
    ax.bar(tools, [rates[tool] * 100 for tool in tools])
    ax.set_ylabel("error rate, %")
    ax.set_title("Tool error rate")
    _save(fig, path)


def plot_verdict_distribution(
    distribution: dict[str, dict[str, int]], path: Path
) -> None:
    """Grouped bars: APPROVE vs REVISE counts, one group per coordination path."""
    fig, ax = plt.subplots()
    paths_ = sorted(distribution)
    verdicts = sorted({v for counts in distribution.values() for v in counts})
    positions = range(len(paths_))
    width = 0.35 if len(verdicts) > 1 else 0.6
    for offset, verdict in enumerate(verdicts):
        values = [distribution[p].get(verdict, 0) for p in paths_]
        shift = (offset - (len(verdicts) - 1) / 2) * width
        ax.bar([pos + shift for pos in positions], values, width, label=verdict)
    ax.set_xticks(list(positions), paths_)
    ax.set_ylabel("verdicts")
    ax.set_title("Verdict distribution by coordination path")
    if verdicts:
        ax.legend()
    _save(fig, path)


def plot_revision_rounds(rounds_per_session: dict[str, int], path: Path) -> None:
    """Bar chart: revision rounds each recorded run went through."""
    fig, ax = plt.subplots()
    sessions = sorted(rounds_per_session)
    labels = [f"run {i + 1}" for i in range(len(sessions))]
    values = [rounds_per_session[session] for session in sessions]
    ax.bar(labels, values)
    ax.set_ylabel("revision rounds")
    ax.set_title("Revision rounds per run")
    _save(fig, path)


def plot_arm_contribution(counts: dict[str, int], path: Path) -> None:
    """Stacked bar: how much of the final top-k came from each arm."""
    fig, ax = plt.subplots()
    labels = [label for label in ARM_LABELS if counts.get(label)]
    values = [counts[label] for label in labels] or [0]
    labels = labels or ["no data"]
    bottom = 0.0
    for label, value in zip(labels, values, strict=True):
        ax.bar(["final top-k"], [value], bottom=bottom, label=label)
        bottom += value
    ax.set_ylabel("documents")
    ax.set_title("Arm contribution to the final answer")
    ax.legend()
    _save(fig, path)


def plot_rerank_effect(pairs: list[tuple[int, int]], path: Path) -> None:
    """Scatter: does the reranker pull documents up from the tail."""
    fig, ax = plt.subplots()
    if pairs:
        before, after = zip(*pairs, strict=True)
        ax.scatter(before, after)
        limit = max(max(before), max(after)) + 1
        ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
    ax.set_xlabel("rank before reranking")
    ax.set_ylabel("rank after reranking")
    ax.set_title("Reranker effect")
    _save(fig, path)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -- report -------------------------------------------------------------


def build_report(settings: Settings, now: datetime | None = None) -> ReportPaths:
    """Read every telemetry stream and write CSV, PNG charts and a report.

    Parameters
    ----------
    settings : Settings
        `output_dir` is where every artifact is written, under the same
        directory the agents' reports go to.
    now : datetime, optional
        Stamps the output filenames. Defaults to the current UTC time;
        passing one explicitly keeps the test suite deterministic.

    Returns
    -------
    ReportPaths
        The Markdown report, the CSV and the directory the charts live in.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d")
    output_dir = paths.resolve(settings.output_dir)
    charts_dir = output_dir / "charts"

    tool_events = read_all_tool_events(settings)
    retrieval_events = read_all_retrieval_events(settings)
    verdict_events = read_all_verdict_events(settings)

    counts_by_agent = tool_call_counts_by_agent(tool_events)
    latency_by_agent = tool_latency_by_agent(tool_events)
    error_rates = tool_error_rates(tool_events)
    distribution = verdict_distribution(verdict_events)
    rounds_per_session = revision_rounds_per_session(verdict_events)
    contribution = arm_contribution(retrieval_events)
    movement = rerank_movement(retrieval_events)

    plot_tool_calls_by_agent(counts_by_agent, charts_dir / CHART_TOOL_CALLS_BY_AGENT)
    plot_tool_latency_by_agent(
        latency_by_agent, charts_dir / CHART_TOOL_LATENCY_BY_AGENT
    )
    plot_tool_error_rates(error_rates, charts_dir / CHART_TOOL_ERRORS)
    plot_verdict_distribution(distribution, charts_dir / CHART_VERDICT_DISTRIBUTION)
    plot_revision_rounds(rounds_per_session, charts_dir / CHART_REVISION_ROUNDS)
    plot_arm_contribution(contribution, charts_dir / CHART_ARM_CONTRIBUTION)
    plot_rerank_effect(movement, charts_dir / CHART_RERANK_EFFECT)

    csv_path = output_dir / f"{stamp}-usage.csv"
    _write_csv(tool_events, csv_path)

    markdown_path = output_dir / f"{stamp}-usage-report.md"
    markdown_path.write_text(
        _render_markdown(
            now,
            tool_events,
            retrieval_events,
            verdict_events,
            counts_by_agent,
            latency_by_agent,
            error_rates,
            distribution,
            rounds_per_session,
            contribution,
        ),
        encoding="utf-8",
    )

    return ReportPaths(markdown=markdown_path, csv=csv_path, charts_dir=charts_dir)


def _write_csv(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["ts", "session", "tool", "agent", "ok", "seconds", "result_chars", "args"]
        )
        for event in events:
            writer.writerow(
                [
                    event.get("ts"),
                    event.get("session"),
                    event.get("tool"),
                    event.get("agent"),
                    event.get("ok"),
                    event.get("seconds"),
                    event.get("result_chars"),
                    json.dumps(event.get("args", {}), ensure_ascii=False),
                ]
            )


def _render_markdown(
    now: datetime,
    tool_events: list[dict[str, Any]],
    retrieval_events: list[dict[str, Any]],
    verdict_events: list[dict[str, Any]],
    counts_by_agent: dict[str, dict[str, int]],
    latency_by_agent: dict[str, tuple[float, float]],
    error_rates: dict[str, float],
    distribution: dict[str, dict[str, int]],
    rounds_per_session: dict[str, int],
    contribution: dict[str, int],
) -> str:
    lines = [
        f"# Usage report - {now.date().isoformat()}",
        "",
        f"{len(tool_events)} tool calls, {len(retrieval_events)} "
        f"knowledge_search retrievals, {len(verdict_events)} Critic verdicts, "
        "across every recorded session.",
        "",
        "## Tool calls by agent",
        f"![tool calls by agent](charts/{CHART_TOOL_CALLS_BY_AGENT})",
        "",
        "## Tool latency by agent (median / p95)",
        f"![tool latency by agent](charts/{CHART_TOOL_LATENCY_BY_AGENT})",
        "",
        "| agent | median s | p95 s |",
        "| --- | --- | --- |",
    ]
    for agent in sorted(latency_by_agent):
        median, p95 = latency_by_agent[agent]
        lines.append(f"| {agent} | {median:.2f} | {p95:.2f} |")

    lines += [
        "",
        "## Tool error rate",
        f"![tool error rate](charts/{CHART_TOOL_ERRORS})",
        "",
        "| tool | error rate |",
        "| --- | --- |",
    ]
    for tool in sorted(error_rates):
        lines.append(f"| {tool} | {error_rates[tool]:.0%} |")

    lines += [
        "",
        "## Verdict distribution by coordination path",
        f"![verdict distribution](charts/{CHART_VERDICT_DISTRIBUTION})",
        "",
        "| path | APPROVE | REVISE |",
        "| --- | --- | --- |",
    ]
    for path_name in sorted(distribution):
        counts = distribution[path_name]
        lines.append(
            f"| {path_name} | {counts.get('APPROVE', 0)} | {counts.get('REVISE', 0)} |"
        )

    lines += [
        "",
        "## Revision rounds per run",
        f"![revision rounds](charts/{CHART_REVISION_ROUNDS})",
        "",
        f"{len(rounds_per_session)} recorded run(s), "
        f"{sum(rounds_per_session.values())} total revision round(s).",
        "",
        "## Retrieval arm contribution",
        f"![arm contribution](charts/{CHART_ARM_CONTRIBUTION})",
        "",
        f"semantic only: {contribution['semantic']}, lexical only: "
        f"{contribution['lexical']}, both arms: {contribution['both']}.",
        "",
        "## Reranker effect",
        f"![reranker effect](charts/{CHART_RERANK_EFFECT})",
        "",
        "Points below the diagonal moved up after reranking.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()
    result = build_report(settings)
    print(f"Report: {result.markdown}")
    print(f"CSV: {result.csv}")
    print(f"Charts: {result.charts_dir}")


if __name__ == "__main__":
    main()
