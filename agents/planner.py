"""Planner sub-agent: turns a user request into a structured `ResearchPlan`.

Bound to `web_search` and `knowledge_search` only -- reconnaissance to
understand the domain, not a deep read of any one source. Stateless per
invocation: no checkpointer, one human message in, one `ResearchPlan` out.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from config import Settings, load_settings
from prompts import build_planner_prompt
from schemas import ResearchPlan
from tools import PLANNER_TOOLS
from tracing import configure_tracing

# A small, fixed reconnaissance budget -- not a Settings field, since the
# experiments this project varies (stage 11) tune the Researcher and Critic
# budgets, not how much the Planner may look around before it decomposes.
PLANNER_TOOL_CALL_LIMIT = 4

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* `strict`, which leaves the provider free to treat the
# schema's `required` list as a hint and return a plan missing a field.
# Naming the strategy explicitly is what puts `"strict": true` on the wire.
PLANNER_RESPONSE_FORMAT = ProviderStrategy(ResearchPlan, strict=True)


def create_planner_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[ResearchPlan], None, InputAgentState, OutputAgentState[ResearchPlan]
]:
    """Build the Planner sub-agent.

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
        A graph whose `invoke` result carries the parsed `ResearchPlan` in
        `result["structured_response"]`.
    """
    settings = settings or load_settings()
    configure_tracing(settings)
    chat_model = model or _build_model(settings)

    return create_agent(
        model=chat_model,
        tools=PLANNER_TOOLS,
        system_prompt=build_planner_prompt(settings.planner_prompt_version),
        response_format=PLANNER_RESPONSE_FORMAT,
        middleware=[ToolCallLimitMiddleware(run_limit=PLANNER_TOOL_CALL_LIMIT)],
    )


def _build_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.planner_model_name or settings.model_name,
        api_key=settings.api_key,
        temperature=settings.temperature,
    )
