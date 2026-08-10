"""Trajectory-level evaluators: does a real run's tool-call sequence match
the reference `plan -> research -> critique -> save_report` trajectory, and
does an LLM judge consider it accurate.

The obvious choice is
`create_trajectory_match_evaluator(trajectory_match_mode="unordered")`, but
probing the installed `agentevals` before trusting it for a real sweep found
`"unordered"` mode is `_is_trajectory_superset` in *both* directions -- with
`tool_args_match_mode` at its own default (`"exact"`), that is a strict
multiset match, no better tolerance for an extra revision round than
`"strict"` mode itself. A real run that gets `REVISE`d once makes two
`research`/`critique` calls instead of one, which is a correct outcome
(acceptance criterion 5 of the assignment), not a deviation from the
reference. `"superset"` mode (only `output ⊇ reference`, checked one
direction) combined with `tool_args_match_mode="ignore"` (call arguments
differ by round -- revision feedback text, updated findings -- and were
never meant to be matched literally) is what actually tolerates this,
confirmed against the installed library with a scripted two-round
trajectory before it replaced the obvious choice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)
from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain_openai import ChatOpenAI

from config import Settings

# The minimal, one-revision-round trajectory every real run must contain, at
# least. Tool call arguments are omitted (`tool_args_match_mode="ignore"`
# below never reads them) -- only the four tool *names*, in this order,
# matter as the floor a real trajectory is checked against.
REFERENCE_TOOL_NAMES: tuple[str, ...] = ("plan", "research", "critique", "save_report")


def _reference_trajectory() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, name in enumerate(REFERENCE_TOOL_NAMES, start=1):
        call_id = f"reference_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "", "tool_call_id": call_id})
    return messages


REFERENCE_TRAJECTORY: list[dict[str, Any]] = _reference_trajectory()

# How each coordination step announces itself on the graph path, where
# `plan`/`research`/`critique` are `StateGraph` nodes rather than tools and
# therefore emit no tool call at all -- only the marker `orchestrator.py`
# writes at the head of each node's own `AIMessage`, plus `save_report`'s
# outcome string. Matched as a prefix, so the round number in
# `[Researcher, round N]` does not need enumerating.
_GRAPH_NODE_MARKERS: tuple[tuple[str, str], ...] = (
    ("[Planner]", "plan"),
    ("[Researcher", "research"),
    ("[Critic]", "critique"),
    ("Report saved to:", "save_report"),
)

_match_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="superset", tool_args_match_mode="ignore"
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content if isinstance(content, str) else ""


def _tool_call_names(message: Any) -> list[str]:
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    names = []
    for call in calls or []:
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
            if name:
                names.append(str(name))
    return names


def observed_tool_names(messages: list[Any]) -> list[str]:
    """Extract the coordination steps a run actually took, on either path.

    Reads real tool calls where there are any (the supervisor path, whose
    `plan`/`research`/`critique` genuinely are tools) and falls back to
    `_GRAPH_NODE_MARKERS` on a message that made no tool call (the graph
    path, whose same three steps are nodes). A message is never counted
    twice: the marker scan only runs when the message called no tool.

    Parameters
    ----------
    messages : list
        `LangChain` message objects or their OpenAI-format dict equivalents.

    Returns
    -------
    list of str
        Step names in the order they occurred, e.g.
        `["plan", "research", "critique", "save_report"]`.
    """
    names: list[str] = []
    for message in messages:
        called = _tool_call_names(message)
        if called:
            names.extend(called)
            continue
        text = _message_text(message)
        for marker, step in _GRAPH_NODE_MARKERS:
            if text.startswith(marker):
                names.append(step)
                break
    return names


def _as_tool_call_trajectory(messages: list[Any]) -> list[dict[str, Any]]:
    """Render `observed_tool_names` back into the shape `agentevals` reads."""
    normalized: list[dict[str, Any]] = []
    for index, name in enumerate(observed_tool_names(messages), start=1):
        call_id = f"observed_{index}"
        normalized.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            }
        )
        normalized.append({"role": "tool", "content": "", "tool_call_id": call_id})
    return normalized


def trajectory_match(
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score whether a run's trajectory contains the reference tool sequence.

    Parameters
    ----------
    outputs : dict
        The end-to-end target's return value; only `outputs["messages"]` is
        read. `reference_outputs` (the dataset row's own `outputs`) is
        accepted for signature compatibility with `evaluate()` but ignored
        -- the reference this evaluator checks against is the fixed
        `REFERENCE_TRAJECTORY`, not anything a dataset row carries.

    Returns
    -------
    dict
        `openevals.types.EvaluatorResult`, structurally: `score` is `True`
        when every reference tool call has a match in the run (extra calls,
        e.g. a second revision round, are allowed).
    """
    return _match_evaluator(
        outputs=_as_tool_call_trajectory(outputs.get("messages", [])),
        reference_outputs=REFERENCE_TRAJECTORY,
    )


def make_trajectory_accuracy_judge(
    settings: Settings,
) -> Callable[..., dict[str, Any]]:
    """Build the LLM-as-judge trajectory evaluator, bound to a real model.

    `create_trajectory_llm_as_judge(model="openai:...")` would let the
    library build its own client via `init_chat_model`, which reads
    `OPENAI_API_KEY` from `os.environ` -- never populated from `.env`
    (confirmed the expensive way: a real sweep failed this on every single
    row, both orchestration paths, with `OpenAIError: Missing credentials`,
    even though `.env` held a valid key). Every agent factory in this
    project avoids the same trap by constructing `ChatOpenAI(api_key=
    settings.api_key, ...)` explicitly rather than relying on the ambient
    environment; passing that same explicit model via `judge=` is what
    makes this evaluator consistent with the rest of the project instead of
    a second, differently-broken way to reach OpenAI.

    Parameters
    ----------
    settings : Settings
        Supplies `api_key` and `model_name` for the judge model.

    Returns
    -------
    callable
        `trajectory_accuracy_judge(*, outputs, ...) -> EvaluatorResult`,
        matching the shape `evaluate()` calls every other evaluator with.
    """
    judge_model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key)
    judge_evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE, judge=judge_model
    )

    def trajectory_accuracy_judge(
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LLM-as-judge accuracy score against the reference trajectory.

        Parameters
        ----------
        outputs : dict
            The end-to-end target's return value; only `outputs["messages"]`
            is read.

        Returns
        -------
        dict
            `openevals.types.EvaluatorResult`, structurally: `agentevals`'
            own `feedback_key="trajectory_accuracy"` score, `True`/`False`
            by default (`create_trajectory_llm_as_judge` was not given
            `continuous=True`).
        """
        return judge_evaluator(
            outputs=outputs.get("messages", []), reference_outputs=REFERENCE_TRAJECTORY
        )

    return trajectory_accuracy_judge
