"""Critic sub-agent: independently verifies research findings and returns a
structured verdict.

Bound to `web_search`, `read_url`, `knowledge_search` -- the same three
tools as the Researcher, so the Critic checks facts through the same
sources rather than only reading the Researcher's own text. Response format
is `CritiqueResult`; `CriticVerificationMiddleware` forces at least one
verification call before a verdict is accepted. Stateless per invocation:
no checkpointer, one human message in, one `CritiqueResult` out.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from config import Settings, load_settings
from middleware import CriticVerificationMiddleware
from prompts import build_critic_prompt
from schemas import CritiqueResult
from tools import CRITIC_TOOLS
from tracing import configure_tracing

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* `strict`, which leaves the provider free to omit
# `verdict` -- the one field the whole revision loop reads. Naming the
# strategy explicitly is what puts `"strict": true` on the wire.
CRITIC_RESPONSE_FORMAT = ProviderStrategy(CritiqueResult, strict=True)


def create_critic_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[CritiqueResult], None, InputAgentState, OutputAgentState[CritiqueResult]
]:
    """Build the Critic sub-agent.

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
        A graph whose `invoke` result carries the parsed `CritiqueResult` in
        `result["structured_response"]`.
    """
    settings = settings or load_settings()
    configure_tracing(settings)
    chat_model = model or _build_model(settings)

    # Annotated explicitly: a bare list literal makes mypy infer the element
    # type from the first entry, which the second middleware then fails to
    # match structurally even though both are valid `AgentMiddleware`s.
    agent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolCallLimitMiddleware(run_limit=settings.critic_max_tool_calls),
        CriticVerificationMiddleware(),
    ]

    graph = create_agent(
        model=chat_model,
        tools=CRITIC_TOOLS,
        system_prompt=build_critic_prompt(
            settings.critic_prompt_version, today=date.today()
        ),
        response_format=CRITIC_RESPONSE_FORMAT,
        middleware=agent_middleware,
    )
    # Read by telemetry.TelemetryHandler off on_tool_start's own metadata --
    # confirmed by a probe script to survive nested invocation, so every
    # tool call this graph's own ReAct loop makes attributes to "critic"
    # regardless of which module invokes it.
    return graph.with_config(metadata={"agent": "critic"}, tags=["agent:critic"])


def _build_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.critic_model_name or settings.model_name,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )
