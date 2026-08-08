"""HITL decision adapter: the assignment's three-choice UX over the real
`HumanInTheLoopMiddleware` decision contract.

The assignment's own `edit` resume payload
(`{"type":"edit","edited_action":{"feedback": ...}}`) does not match `Action`
(`TypedDict{name, args}`) and raises `KeyError: 'name'` if sent to
`HumanInTheLoopMiddleware._process_decision` as-is. `edit` is mapped instead
onto a real `reject` decision carrying the user's feedback as the message:
the tool call does not execute, the model receives an error `ToolMessage`
with that feedback plus an instruction to revise and call `save_report`
again -- exactly the behaviour the assignment describes in prose, built on
top of the real decision types (`approve`, `edit`, `reject`, `respond`).

Application layer: `orchestrator.py` (stage 7) imports `build_interrupt_request`
for the graph path's `write_node`, so `hitl.py` must never import `main` --
application must not import interface.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    ApproveDecision,
    Decision,
    DecisionType,
    HITLRequest,
    RejectDecision,
    ReviewConfig,
)

ReplChoice = Literal["approve", "edit", "reject"]

_ARG_PREVIEW_LENGTH = 300
_EDIT_INSTRUCTION = "Revise the report and call save_report again."


def build_interrupt_request(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    allowed_decisions: tuple[DecisionType, ...] = ("approve", "edit", "reject"),
    description: str | None = None,
) -> HITLRequest:
    """Build the interrupt payload `HumanInTheLoopMiddleware` would send.

    Parameters
    ----------
    tool_name : str
        Name of the tool call awaiting approval.
    tool_args : dict
        The tool call's arguments.
    allowed_decisions : tuple of DecisionType, default ("approve", "edit", "reject")
        Decisions the REPL offers for this action.
    description : str, optional
        Human-readable description. Defaults to a plain "Tool / Args" line.

    Returns
    -------
    HITLRequest
        Exactly the shape `HumanInTheLoopMiddleware.after_model` constructs,
        so a caller building this payload by hand (the graph path's
        `write_node`) and the middleware's own construction are
        indistinguishable to `main.py`'s one interrupt handler.
    """
    action_request = ActionRequest(
        name=tool_name,
        args=tool_args,
        description=description or f"Tool: {tool_name}\nArgs: {tool_args}",
    )
    review_config = ReviewConfig(
        action_name=tool_name, allowed_decisions=list(allowed_decisions)
    )
    return HITLRequest(action_requests=[action_request], review_configs=[review_config])


def build_decisions(choice: ReplChoice, text: str = "") -> list[Decision]:
    """Map one REPL choice onto the decisions `Command(resume=...)` expects.

    Parameters
    ----------
    choice : {"approve", "edit", "reject"}
        The user's choice at the approval prompt.
    text : str, default ""
        Feedback (for `edit`) or a reason (for `reject`). Ignored for
        `approve`.

    Returns
    -------
    list of Decision
        Always one decision -- this project gates exactly one tool
        (`save_report`) at a time, so one interrupted tool call needs one
        decision.

    Raises
    ------
    ValueError
        If `choice` is not one of the three supported values.
    """
    if choice == "approve":
        return [ApproveDecision(type="approve")]
    if choice == "edit":
        feedback = text.strip()
        message = f"{feedback} {_EDIT_INSTRUCTION}" if feedback else _EDIT_INSTRUCTION
        return [RejectDecision(type="reject", message=message)]
    if choice == "reject":
        reason = text.strip()
        decision: RejectDecision = {"type": "reject"}
        if reason:
            decision["message"] = reason
        return [decision]
    raise ValueError(f"Unknown HITL choice: {choice!r}")


def render_interrupt(request: HITLRequest) -> str:
    """Render an interrupt payload as a console preview.

    Parameters
    ----------
    request : HITLRequest
        The pending interrupt, as delivered on `Interrupt.value`.

    Returns
    -------
    str
        One block per action request: the tool name and each argument,
        long values truncated to a preview.
    """
    return "\n\n".join(_render_action(action) for action in request["action_requests"])


def _render_action(action: ActionRequest) -> str:
    lines = [f"Tool:  {action['name']}"]
    lines.extend(f"  {key}: {_preview(value)}" for key, value in action["args"].items())
    return "\n".join(lines)


def _preview(value: object) -> str:
    text = str(value)
    if len(text) <= _ARG_PREVIEW_LENGTH:
        return text
    return f"{text[:_ARG_PREVIEW_LENGTH]}..."
