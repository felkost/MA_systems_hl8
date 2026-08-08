"""Explicit `StateGraph` orchestrator: the second coordination path.

Same Plan -> Research -> Critique -> write loop as `supervisor.py`'s
agent-as-tool Supervisor, built instead as an explicit graph with a
`revision_round` counter in state and `add_conditional_edges` routing on the
Critic's verdict. The revision cap is state arithmetic here, not
`ToolCallLimitMiddleware` -- the two mechanisms are meant to enforce the
identical `Settings.max_revisions` bound independently, on two different
paths, not to share code with each other.

`write_node` builds its interrupt payload with `hitl.build_interrupt_request`,
the same shape `HumanInTheLoopMiddleware.after_model` produces on the
supervisor path -- this is what lets `main.py` drive both paths with one
unchanged interrupt handler.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

import hitl
import telemetry
from agents.critic import create_critic_agent
from agents.planner import create_planner_agent
from agents.research import create_research_agent
from config import Settings, load_settings
from schemas import render_critique, render_plan
from tools import save_report
from tracing import configure_tracing

# Duplicated from `supervisor.py` rather than imported: the two orchestration
# paths are meant to stay independent alternatives, not share a coupling
# neither needs for its own sake.
_CRITIQUE_INPUT_TEMPLATE = "Original request: {request}\n\nFindings:\n{findings}"


class ResearchState(TypedDict, total=False):
    """Graph state for the explicit orchestration path.

    Every field past `messages` is filled in progressively by one node each,
    which is why the class is `total=False` -- a node reads only the fields
    the nodes before it are guaranteed to have set.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    findings: str
    critique: str
    verdict: Literal["APPROVE", "REVISE"]
    revision_round: int
    report: str


def create_orchestrator(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
    """Build the explicit `StateGraph` orchestrator.

    Parameters
    ----------
    settings : Settings, optional
        Configuration to build the graph from. Loaded from the environment
        when omitted.
    model : BaseChatModel, optional
        Chat model forwarded to every sub-agent factory in place of the one
        built from `settings`. Tests inject a scripted fake here, or
        monkeypatch the `create_*_agent` names this module calls directly.

    Returns
    -------
    CompiledStateGraph
        `START -> plan -> research -> critique -> {research | write} -> END`,
        with an `InMemorySaver` checkpointer -- required for `write_node`'s
        `interrupt()` call to survive to the matching resume call, exactly
        as the supervisor path needs one for its `HumanInTheLoopMiddleware`.
    """
    settings = settings or load_settings()
    configure_tracing(settings)

    def _log_verdict(config: RunnableConfig, verdict: str, round_number: int) -> None:
        """Record a verdict under this run's own `thread_id`.

        A direct `graph.invoke(...)` in a test, without a `thread_id` in
        `config`, has nothing to namespace a verdict log under -- it simply
        logs nothing rather than raising or falling back to a shared file.

        `load_settings()` is called here rather than reusing the `settings`
        this factory closed over, because `config.override_output_dir` is a
        `ContextVar` entered *per run* -- an evaluation sweep enters it long
        after the graph was compiled. A captured `settings` would send every
        concurrent example's verdicts to the directory that was current at
        build time; `supervisor.py::_log_verdict` already resolves it this
        way for the same reason.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return
        telemetry.log_verdict(
            load_settings(),
            thread_id,
            path="orchestrator",
            verdict=verdict,
            round=round_number,
        )

    def plan_node(state: ResearchState) -> dict[str, Any]:
        request_text = _first_human_text(state["messages"])
        result = create_planner_agent(settings, model=model).invoke(
            {"messages": [HumanMessage(request_text)]}
        )
        rendered = render_plan(result["structured_response"])
        return {
            "plan": rendered,
            "messages": [AIMessage(content=f"[Planner]\n{rendered}")],
        }

    def research_node(state: ResearchState) -> dict[str, Any]:
        revision_round = state.get("revision_round", 0)
        if revision_round == 0:
            request_text = state["plan"]
        else:
            request_text = (
                "Revise the research based on this critique.\n\n"
                f"{state['critique']}\n\nPrior findings:\n{state['findings']}"
            )
        result = create_research_agent(settings, model=model).invoke(
            {"messages": [HumanMessage(request_text)]}
        )
        findings = str(result["messages"][-1].text)
        return {
            "findings": findings,
            "messages": [
                AIMessage(
                    content=f"[Researcher, round {revision_round + 1}]\n{findings}"
                )
            ],
        }

    def critique_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
        request_text = _first_human_text(state["messages"])
        result = create_critic_agent(settings, model=model).invoke(
            {
                "messages": [
                    HumanMessage(
                        _CRITIQUE_INPUT_TEMPLATE.format(
                            request=request_text, findings=state["findings"]
                        )
                    )
                ]
            }
        )
        structured = result["structured_response"]
        rendered = render_critique(structured)
        _log_verdict(config, structured.verdict, state.get("revision_round", 0))
        update: dict[str, Any] = {
            "critique": rendered,
            "verdict": structured.verdict,
            "messages": [AIMessage(content=f"[Critic]\n{rendered}")],
        }
        if structured.verdict == "REVISE":
            update["revision_round"] = state.get("revision_round", 0) + 1
        return update

    def write_node(state: ResearchState) -> dict[str, Any]:
        filename, content = _compose_report(state)
        request = hitl.build_interrupt_request(
            "save_report", {"filename": filename, "content": content}
        )
        resume_value = cast("dict[str, list[dict[str, Any]]]", interrupt(request))
        decision = resume_value["decisions"][0]
        if decision["type"] == "approve":
            outcome = str(
                save_report.invoke({"filename": filename, "content": content})
            )
        else:
            reason = decision.get("message", "")
            outcome = f"Report not saved: {reason}" if reason else "Report not saved."
        return {"report": outcome, "messages": [AIMessage(content=outcome)]}

    def _route_after_critique(state: ResearchState) -> Literal["research", "write"]:
        if state.get("verdict") == "APPROVE":
            return "write"
        assert settings is not None
        if state.get("revision_round", 0) <= settings.max_revisions:
            return "research"
        return "write"

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("critique", critique_node)
    graph.add_node("write", write_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "critique")
    graph.add_conditional_edges(
        "critique", _route_after_critique, {"research": "research", "write": "write"}
    )
    graph.add_edge("write", END)

    compiled = graph.compile(checkpointer=InMemorySaver())
    # Read by telemetry.TelemetryHandler off on_tool_start's own metadata --
    # tags write_node's own save_report call (not itself a tagged sub-agent
    # graph) as "orchestrator"; each node's own sub-agent invocation carries
    # its own "agent" tag that wins for its own nested tool calls (confirmed
    # by a probe script).
    return compiled.with_config(
        metadata={"agent": "orchestrator"}, tags=["agent:orchestrator"]
    )


def _first_human_text(messages: list[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


def _compose_report(state: ResearchState) -> tuple[str, str]:
    request_text = _first_human_text(state["messages"])
    content = (
        f"# {request_text}\n\n"
        f"{state.get('findings', '')}\n\n"
        f"## Critique\n\n{state.get('critique', '')}"
    )
    return f"{_slugify(request_text)}.md", content


def _slugify(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return "-".join(words[:8]) or "report"
