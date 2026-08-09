"""Supervisor: coordinates the Planner, Researcher and Critic sub-agents.

Each sub-agent is wrapped as an agent-as-tool function -- `plan`, `research`,
`critique` -- so the Supervisor's own `create_agent` graph calls them the
same way it calls `save_report`. The wrappers return rendered markdown
(`render_plan`/`render_critique`), never a sub-agent's raw structured
response: both the Planner and the Critic resolve to `ProviderStrategy` on
the real OpenAI path, where the last message *is* raw JSON, and the
Supervisor must not see that.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

import telemetry
from agents.critic import create_critic_agent
from agents.planner import create_planner_agent
from agents.research import create_research_agent
from config import Settings, load_settings
from prompts import build_supervisor_prompt
from schemas import RESEARCH_INPUT_TEMPLATE, render_critique, render_plan
from tools import SUPERVISOR_TOOLS
from tracing import configure_tracing

CRITIQUE_INPUT_TEMPLATE = "Original request: {request}\n\nFindings:\n{findings}"

# A sub-agent failure is data the Supervisor reads and reacts to, the same
# way every tool in this project reports failure. The underlying exception
# text is withheld: it can carry a provider payload, a local path or a raw
# model response, and it would reach the model unchanged.
_SUB_AGENT_ERROR = (
    "ERROR: The {agent} sub-agent failed to produce a usable result. "
    "Do not call {tool} again with the same input -- either continue with "
    "what you already have, or call it once more with a simpler, more "
    "specific request."
)


@tool
def plan(request: str) -> str:
    """Decompose a user's research request into a structured research plan.

    Call this first, before research or critique -- it turns a request into
    concrete search queries, source selection and an output format.
    """
    try:
        result = create_planner_agent(load_settings()).invoke(
            {"messages": [HumanMessage(request)]}
        )
        return render_plan(result["structured_response"])
    except Exception:
        return _SUB_AGENT_ERROR.format(agent="Planner", tool="plan")


@tool
def research(request: str, runtime: ToolRuntime) -> str:
    """Execute a research plan, or a revision request, and return findings.

    Pass the rendered plan on the first call, and the critic's
    revision_requests as feedback on any later call.
    """
    original_request = _first_human_text(runtime.state["messages"])
    try:
        result = create_research_agent(load_settings()).invoke(
            {
                "messages": [
                    HumanMessage(
                        RESEARCH_INPUT_TEMPLATE.format(
                            request=original_request, task=request
                        )
                    )
                ]
            }
        )
        return str(result["messages"][-1].text)
    except Exception:
        return _SUB_AGENT_ERROR.format(agent="Researcher", tool="research")


@tool
def critique(findings: str, runtime: ToolRuntime) -> str:
    """Independently verify research findings and return a structured verdict.

    Call this after research, with the findings it returned.
    """
    original_request = _first_human_text(runtime.state["messages"])
    try:
        result = create_critic_agent(load_settings()).invoke(
            {
                "messages": [
                    HumanMessage(
                        CRITIQUE_INPUT_TEMPLATE.format(
                            request=original_request, findings=findings
                        )
                    )
                ]
            }
        )
        structured = result["structured_response"]
        _log_verdict(runtime, structured.verdict)
        return render_critique(structured)
    except Exception:
        return _SUB_AGENT_ERROR.format(agent="Critic", tool="critique")


def _first_human_text(messages: list[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


def _log_verdict(runtime: ToolRuntime, verdict: str) -> None:
    """Record the Critic's verdict, namespaced under this run's `thread_id`.

    A `critique` call made outside a real graph run -- most of this
    project's own tests, and any direct tool invocation -- carries no
    `thread_id` in `runtime.config`, and must not raise or fall back to
    some shared file; it simply logs nothing.
    """
    thread_id = runtime.config.get("configurable", {}).get("thread_id")
    if thread_id is None:
        return
    round_number = max(0, _prior_critique_calls(runtime.state["messages"]) - 1)
    telemetry.log_verdict(
        load_settings(),
        thread_id,
        path="supervisor",
        verdict=verdict,
        round=round_number,
    )


def _prior_critique_calls(messages: list[BaseMessage]) -> int:
    """Count `critique` tool calls already in the conversation.

    Includes the call currently executing: a `ToolNode` appends the
    triggering `AIMessage` to state before running the tool it names, so by
    the time this function runs, that call is already counted -- callers
    subtract one to get a 0-indexed round number.
    """
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "critique"
    )


def create_supervisor(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Supervisor agent.

    Parameters
    ----------
    settings : Settings, optional
        Configuration to build the agent from. Loaded from the environment
        when omitted.
    model : BaseChatModel, optional
        Chat model to use instead of the one built from `settings`. Tests
        inject a scripted fake here instead of reaching OpenAI.

    Returns
    -------
    CompiledStateGraph
        A graph bound to exactly `[plan, research, critique, save_report]`,
        with the revision-round cap enforced by
        `ToolCallLimitMiddleware(tool_name=..., run_limit=1 +
        settings.max_revisions, exit_behavior="continue")` on both
        `research` and `critique`, `save_report` gated behind
        `HumanInTheLoopMiddleware`, and an `InMemorySaver` checkpointer --
        required for the interrupt to survive to the matching resume call.
    """
    settings = settings or load_settings()
    configure_tracing(settings)
    chat_model = model or _build_model(settings)

    # Annotated explicitly: a bare list literal makes mypy infer the element
    # type from the first entry, which the second middleware then fails to
    # match structurally even though both are valid `AgentMiddleware`s.
    agent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolCallLimitMiddleware(
            tool_name="research",
            run_limit=1 + settings.max_revisions,
            exit_behavior="continue",
        ),
        ToolCallLimitMiddleware(
            tool_name="critique",
            run_limit=1 + settings.max_revisions,
            exit_behavior="continue",
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_report": {"allowed_decisions": ["approve", "edit", "reject"]}
            }
        ),
    ]

    graph = create_agent(
        model=chat_model,
        tools=[plan, research, critique, *SUPERVISOR_TOOLS],
        system_prompt=build_supervisor_prompt(settings.supervisor_prompt_version),
        middleware=agent_middleware,
        checkpointer=InMemorySaver(),
    )
    # Read by telemetry.TelemetryHandler off on_tool_start's own metadata --
    # tags the Supervisor's own top-level tool calls (plan/research/critique/
    # save_report) as "supervisor", distinct from the sub-agent each wrapper
    # invokes, which carries its own "agent" tag that wins for its own
    # nested tool calls (confirmed by a probe script).
    return graph.with_config(
        metadata={"agent": "supervisor"}, tags=["agent:supervisor"]
    )


def _build_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )
