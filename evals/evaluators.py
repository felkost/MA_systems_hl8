"""Per-example evaluators for both the end-to-end dataset
(`evals/dataset.py`) and the sub-agent-only dataset
(`evals/subagent_dataset.py`).

Every evaluator returns `score: bool | None`, never folding "not
applicable" into `0`/`False` -- a run whose trajectory has no critique in it
at all (an error, or a dataset row this evaluator was never meant to score)
must show up in a LangSmith experiment as a blank cell, not as a
measured failure. `_mean_excluding_none` is the pinned utility that keeps
that true wherever an evaluator combines more than one applicable-or-not
sub-check into a single score.

`plan_is_structured`, `critic_verified` and `hitl_respected` read either
outputs shape (the end-to-end target's `messages`/`tools_log`, or a
sub-agent target's own structured `plan`/`critique`/`messages`) so the same
function can appear in both `evals/run_eval.py::END_TO_END_EVALUATORS` and
its per-sub-agent evaluator lists without duplicating logic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

_VERIFICATION_TOOLS = frozenset({"web_search", "read_url", "knowledge_search"})
_PLAN_FIELDS = ("goal", "search_queries", "sources_to_check", "output_format")
_PLAN_HEADERS = (
    "**Goal:**",
    "**Search queries:**",
    "**Sources to check:**",
    "**Output format:**",
)
_VERDICT_PATTERN = re.compile(r"\*\*Verdict:\*\* (\w+)")


def _mean_excluding_none(scores: Sequence[float | None]) -> float | None:
    """Average the applicable scores; `None` entries are excluded, not zeroed.

    Parameters
    ----------
    scores : sequence of float or None
        Sub-scores from one or more checks. `None` means "not applicable",
        not "failed".

    Returns
    -------
    float or None
        The mean of the non-`None` entries, or `None` if every entry was.
    """
    applicable = [score for score in scores if score is not None]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL telemetry stream, tolerating a corrupted line.

    A real sweep hit `JSONDecodeError` on a `*-tools.jsonl` line -- almost
    certainly an interleaved write from `ToolNode` running several tool
    calls of one turn concurrently, each through its own
    `TelemetryHandler.on_tool_end` call (a race in already-merged
    `telemetry.py`, flagged separately rather than fixed here). One bad
    line must not crash the evaluator and discard every other event in the
    file the way raising outright would.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _last_containing(messages: Iterable[BaseMessage], marker: str) -> str | None:
    for message in reversed(list(messages)):
        text = _message_text(message)
        if marker in text:
            return text
    return None


def _result(
    key: str, score: bool | float | None, comment: str | None
) -> dict[str, Any]:
    return {"key": key, "score": score, "comment": comment, "metadata": None}


def plan_is_structured(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether a `ResearchPlan` has every field the schema requires filled in.

    Reads `outputs["plan"]` directly for a sub-agent-only row, or scans the
    end-to-end trajectory's messages for `render_plan`'s own markdown
    headers (`schemas.py`) when there is no structured object to read.
    """
    if "plan" in outputs:
        plan = outputs["plan"]
        complete = all(bool(plan.get(field)) for field in _PLAN_FIELDS)
        return _result("plan_is_structured", complete, None)

    text = _last_containing(outputs.get("messages", []), "**Goal:**")
    if text is None:
        return _result("plan_is_structured", None, "no plan found in the trajectory")
    complete = all(header in text for header in _PLAN_HEADERS)
    return _result("plan_is_structured", complete, None)


def plan_covers_request(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether `sources_to_check` matches the row's `expected_sources`.

    Sub-agent-only planner rows carry `reference_outputs["expected_sources"]`
    (`"web"`, `"knowledge_base"`, or `"both"`); a row without one, or an
    `outputs` shape without a structured `plan`, is not applicable.
    """
    plan = outputs.get("plan")
    expected = (reference_outputs or {}).get("expected_sources")
    if plan is None or expected is None:
        return _result("plan_covers_request", None, "no structured plan or expectation")

    sources = {
        str(source).strip().lower() for source in plan.get("sources_to_check", [])
    }
    if expected == "both":
        covers = {"web", "knowledge_base"} <= sources or "both" in sources
    else:
        covers = expected in sources or "both" in sources
    return _result("plan_covers_request", covers, f"sources_to_check={sorted(sources)}")


def critic_verified(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether the Critic called a verification tool before it verdicted.

    `CriticVerificationMiddleware` (`middleware.py`) already forces this at
    the agent level; this evaluator is the independent, telemetry-grounded
    measurement Risk 2 (plan) calls for -- "if measurement still shows
    blanket approval, the honest report says the evaluator-optimizer loop
    did not fire." End-to-end rows read `outputs["tools_log"]`, filtered to
    `agent == "critic"`; sub-agent-only critic rows read `outputs["messages"]`
    directly, since a lone `create_critic_agent` call carries no telemetry
    callback.
    """
    tools_log = outputs.get("tools_log")
    if tools_log is not None:
        events = _read_jsonl(tools_log)
        verified = any(
            event.get("agent") == "critic" and event.get("tool") in _VERIFICATION_TOOLS
            for event in events
        )
        return _result("critic_verified", verified, None)

    messages = outputs.get("messages")
    if messages is None:
        return _result("critic_verified", None, "no trajectory available")
    verified = any(
        isinstance(message, AIMessage)
        and any(call["name"] in _VERIFICATION_TOOLS for call in message.tool_calls)
        for message in messages
    )
    return _result("critic_verified", verified, None)


def _parse_critique(text: str) -> dict[str, Any] | None:
    match = _VERDICT_PATTERN.search(text)
    if match is None:
        return None
    all_dimensions_true = (
        "Fresh:** True" in text
        and "Complete:** True" in text
        and "Well-structured:** True" in text
    )
    gaps_and_requests_empty = (
        "**Gaps:**\n- none" in text and "**Revision requests:**\n- none" in text
    )
    return {
        "verdict": match.group(1),
        "all_dimensions_true": all_dimensions_true,
        "gaps_and_requests_empty": gaps_and_requests_empty,
    }


def verdict_is_justified(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether the Critic's verdict is consistent with its own dimensions.

    The direct measurement for the standing Open Question
    (`insights.md`, "Critic verdict vs. its own dimensions"): `c1` requires
    "APPROVE only when all three [dimensions] hold", a necessary but not a
    sufficient condition, so a `REVISE` with all three `True` and empty
    `gaps`/`revision_requests` is formally allowed by the prompt and wrong
    in practice. This evaluator checks the biconditional `c1` never states
    (`APPROVE` iff all three dimensions are `True` and `gaps`/
    `revision_requests` are both empty), and separately -- when the dataset
    row states one -- whether the verdict matches `expected_verdict`.
    """
    text = _last_containing(outputs.get("messages", []), "**Verdict:**")
    if text is None:
        rendered = outputs.get("rendered", "")
        text = rendered if "**Verdict:**" in rendered else None
    parsed = _parse_critique(text) if text else None
    if parsed is None:
        return _result(
            "verdict_is_justified", None, "no critique found in the trajectory"
        )

    consistent = (
        parsed["verdict"] == "APPROVE"
        and parsed["all_dimensions_true"]
        and parsed["gaps_and_requests_empty"]
    ) or (
        parsed["verdict"] == "REVISE"
        and not (parsed["all_dimensions_true"] and parsed["gaps_and_requests_empty"])
    )

    expected = (reference_outputs or {}).get("expected_verdict")
    matches_expected = None if expected is None else parsed["verdict"] == expected

    score = _mean_excluding_none(
        [
            float(consistent),
            None if matches_expected is None else float(matches_expected),
        ]
    )
    comment = f"verdict={parsed['verdict']} consistent={consistent}"
    if matches_expected is not None:
        comment += f" matches_expected={matches_expected}"
    return _result("verdict_is_justified", score, comment)


def revision_converged(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether a revision loop ended in `APPROVE` or a deliberate cap-out.

    Reads `outputs["verdicts_log"]` -- only the end-to-end target produces
    one, so a sub-agent-only row is not applicable. Not applicable, either,
    to a row whose run never reached `critique` at all.
    """
    verdicts_log = outputs.get("verdicts_log")
    events = _read_jsonl(verdicts_log) if verdicts_log else []
    if not events:
        return _result("revision_converged", None, "no verdicts recorded for this run")

    last = events[-1]
    max_revisions = outputs.get("max_revisions")
    capped_out = max_revisions is not None and last["round"] >= max_revisions
    converged = last["verdict"] == "APPROVE" or capped_out
    return _result(
        "revision_converged",
        converged,
        f"last verdict={last['verdict']} round={last['round']}",
    )


def hitl_respected(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whether a report actually reached disk only through the gated write.

    There is no code path where `save_report` executes without first
    raising the `interrupt()` this project's auto-approve eval harness
    resolves (`HumanInTheLoopMiddleware` / `write_node`), so a non-empty
    `.md` file under `outputs["output_dir"]` is itself the evidence the gate
    was exercised and approved -- not merely that a Supervisor decided to
    call the tool.
    """
    output_dir = outputs.get("output_dir")
    if not output_dir:
        return _result("hitl_respected", None, "no output_dir in outputs")

    saved_reports = list(Path(output_dir).glob("*.md"))
    if not saved_reports:
        attempted = any(
            call.get("name") == "save_report"
            for message in outputs.get("messages", [])
            for call in getattr(message, "tool_calls", [])
        )
        if not attempted:
            return _result(
                "hitl_respected", None, "no save_report call in this trajectory"
            )
        return _result(
            "hitl_respected", False, "save_report attempted but no file written"
        )

    saved = saved_reports[0]
    non_empty = saved.stat().st_size > 0
    return _result("hitl_respected", non_empty, f"saved {saved.name}")
