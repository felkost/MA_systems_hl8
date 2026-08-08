"""Middleware shared by the sub-agents.

Each `create_*_agent` factory wires only the middleware its architecture row
prescribes -- nothing here is applied unconditionally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from prompts import CRITIC_VERIFICATION_INSTRUCTION

_VERIFICATION_TOOLS = frozenset({"web_search", "read_url", "knowledge_search"})


def _run_tool_call_ids(messages: list[BaseMessage], tool_name: str) -> list[str]:
    """Ids of every call to `tool_name` since the most recent `HumanMessage`.

    A limit scoped to "this run" must reset each turn instead of
    accumulating across a checkpointed thread's whole history -- counting
    from the end of `messages` back to the most recent `HumanMessage` is
    what gives a limit that scope.

    Parameters
    ----------
    messages : list of BaseMessage
        The agent state's message list.
    tool_name : str
        The tool to count calls for.

    Returns
    -------
    list of str
        Tool-call ids, oldest first.
    """
    ids: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            ids = []
        elif isinstance(message, AIMessage):
            ids.extend(
                call["id"]
                for call in message.tool_calls
                if call["name"] == tool_name and call["id"] is not None
            )
    return ids


class ReadUrlCapMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Caps how many `read_url` calls the Researcher may make in one run.

    Without a cap the Researcher can spend its whole tool budget reading
    pages a search already found, instead of running the fresh searches the
    plan actually asks for. `max_calls=None` removes the cap.
    """

    def __init__(self, max_calls: int | None) -> None:
        super().__init__()
        self.max_calls = max_calls

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if self.max_calls is None or request.tool_call["name"] != "read_url":
            return handler(request)

        call_id = request.tool_call["id"]
        prior_calls = [
            prior_id
            for prior_id in _run_tool_call_ids(request.state["messages"], "read_url")
            if prior_id != call_id
        ]
        if len(prior_calls) >= self.max_calls:
            return ToolMessage(
                content=(
                    f"ERROR: read_url call limit ({self.max_calls}) reached for "
                    "this run. Run a new web_search or knowledge_search instead "
                    "of reading another page."
                ),
                tool_call_id=call_id,
                name="read_url",
                status="error",
            )
        return handler(request)


class CriticVerificationMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Forces the Critic to verify at least one claim before it verdicts.

    `response_format=CritiqueResult` lets the model end the turn with a
    verdict and no verification call at all -- under `ProviderStrategy` that
    is a message with no tool calls, under `ToolStrategy` (what a fake model
    without provider-strategy support resolves to, e.g. in tests) it is a
    tool call to the synthetic structured-output tool, which is not a call
    to `web_search`/`read_url`/`knowledge_search` either. Either way, if
    none of those three tools ran earlier this turn, this middleware re-runs
    the model call once with `CRITIC_VERIFICATION_INSTRUCTION` appended. The
    retried response is returned as-is, whatever it contains -- one-shot,
    the same shape as `ReadUrlCapMiddleware`'s "since the last
    `HumanMessage`" scoping, so a model that skips verification twice in a
    row cannot make this middleware loop.
    """

    def __init__(self, min_verification_calls: int = 1) -> None:
        super().__init__()
        self.min_verification_calls = min_verification_calls

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        response = handler(request)
        if self._calls_a_verification_tool(response) or self._verified_earlier(request):
            return response

        retry_request = request.override(
            messages=[
                *request.messages,
                HumanMessage(content=CRITIC_VERIFICATION_INSTRUCTION),
            ]
        )
        return handler(retry_request)

    def _verified_earlier(self, request: ModelRequest[ContextT]) -> bool:
        # `AgentState["messages"]` is `list[AnyMessage]`, a Union alias;
        # `list` is invariant, so mypy rejects it as a `list[BaseMessage]`
        # argument even though every member of the union is one.
        messages = cast("list[BaseMessage]", request.state["messages"])
        verified_calls = sum(
            len(_run_tool_call_ids(messages, tool_name))
            for tool_name in _VERIFICATION_TOOLS
        )
        return verified_calls >= self.min_verification_calls

    @staticmethod
    def _calls_a_verification_tool(response: ModelResponse[ResponseT]) -> bool:
        return any(
            isinstance(message, AIMessage)
            and any(call["name"] in _VERIFICATION_TOOLS for call in message.tool_calls)
            for message in response.result
        )
