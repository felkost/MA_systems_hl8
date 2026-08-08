"""Research sub-agent: executes a plan with web, page-read and knowledge-base tools.

Bound to `web_search`, `read_url`, `knowledge_search` and `graph_search`.
Returns free-text findings as structured Markdown with inline citations --
it does not save anything; only the Supervisor's `save_report` tool, gated
by human approval, writes to disk. Stateless per invocation: no
checkpointer, one human message in, one findings message out.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from config import Settings, load_settings
from middleware import ReadUrlCapMiddleware
from prompts import build_researcher_prompt
from tools import RESEARCHER_TOOLS
from tracing import configure_tracing


def create_research_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Research sub-agent.

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
        A graph whose `invoke` result carries the findings as free-text
        Markdown in the last message -- there is no `response_format`.
    """
    settings = settings or load_settings()
    configure_tracing(settings)
    chat_model = model or _build_model(settings)

    # Annotated explicitly: a bare list literal makes mypy infer the element
    # type from the first entry, which the second middleware then fails to
    # match structurally even though both are valid `AgentMiddleware`s.
    agent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolCallLimitMiddleware(run_limit=settings.researcher_max_tool_calls),
        ReadUrlCapMiddleware(settings.max_read_url_per_search),
    ]

    graph = create_agent(
        model=chat_model,
        tools=RESEARCHER_TOOLS,
        system_prompt=build_researcher_prompt(settings.researcher_prompt_version),
        middleware=agent_middleware,
    )
    # Read by telemetry.TelemetryHandler off on_tool_start's own metadata --
    # confirmed by a probe script to survive nested invocation, so every
    # tool call this graph's own ReAct loop makes attributes to "researcher"
    # regardless of which module invokes it.
    return graph.with_config(
        metadata={"agent": "researcher"}, tags=["agent:researcher"]
    )


def _build_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )
