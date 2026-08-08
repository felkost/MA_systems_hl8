"""REPL entry point: streams the Supervisor's responses and drives the
human-in-the-loop interrupt/resume loop around `save_report`.

One `thread_id` per process. The Supervisor's `InMemorySaver` checkpointer
only remembers a conversation while `thread_id` stays the same, so the REPL
builds it once at startup and reuses the same `config` for every question
asked in the session, including every interrupt/resume cycle inside it.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Iterator
from typing import Any, cast

from langchain.agents.middleware.human_in_the_loop import Decision
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

import hitl
from config import Settings, apply_overrides, load_settings
from orchestrator import create_orchestrator
from retriever import IndexMismatchError, get_retriever
from supervisor import create_supervisor

_ICONS_ASCII = {"tool": "[tool]", "result": "[result]"}
_ICONS_UTF8 = {"tool": "\U0001f527", "result": "\U0001f4ce"}
_icons = _ICONS_ASCII

_RESULT_PREVIEW_LENGTH = 300
_APPROVAL_BANNER = "=" * 60


def configure_console() -> None:
    """Reconfigure stdout/stderr to UTF-8, or keep the plain ASCII icons.

    Emoji in `print()` raise `UnicodeEncodeError` on a Windows console stuck
    on a non-UTF-8 code page -- confirmed once already in this project, not
    hypothetical. Reconfiguring first is what makes the emoji icons safe;
    keeping the ASCII fallback is what keeps the REPL usable when a stream
    cannot be reconfigured at all.
    """
    global _icons
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            return
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            return
    _icons = _ICONS_UTF8


def build_run_config(thread_id: str, settings: Settings) -> RunnableConfig:
    """Build the run config for one REPL session.

    Parameters
    ----------
    thread_id : str
        Identifies the conversation to the checkpointer. Must stay the same
        for the whole session -- a new value starts a new, empty memory.
    settings : Settings
        Supplies `recursion_limit`.

    Returns
    -------
    RunnableConfig
        `{"configurable": {"thread_id": ...}, "recursion_limit": ...}`.
    """
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=settings.recursion_limit,
    )


def format_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render one tool call for the console."""
    rendered_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{_icons['tool']} {name}({rendered_args})"


def format_tool_result(content: str, *, limit: int = _RESULT_PREVIEW_LENGTH) -> str:
    """Render one tool result for the console, truncated to `limit`."""
    text = content if len(content) <= limit else f"{content[:limit]}..."
    return f"{_icons['result']} {text}"


def resume_turn(
    graph: Any,
    config: RunnableConfig,
    decisions: list[Decision],
    *,
    thread_id: str,
) -> Iterator[dict[str, Any]]:
    """Resume an interrupted run with the given decisions.

    Parameters
    ----------
    graph : object with a `.stream` method
        The compiled Supervisor graph.
    config : RunnableConfig
        The run config to resume with.
    decisions : list of Decision
        Built by `hitl.build_decisions`.
    thread_id : str
        This session's own thread id, compared against `config`'s.

    Returns
    -------
    Iterator of dict
        The resumed run's `stream_mode="updates"` chunks.

    Raises
    ------
    ValueError
        If `config` carries a different `thread_id` -- `InMemorySaver` has
        no pending interrupt for a thread it never ran, so resuming it would
        silently start a brand-new, empty conversation instead of
        continuing the one the human is answering.
    """
    actual_thread_id = config.get("configurable", {}).get("thread_id")
    if actual_thread_id != thread_id:
        raise ValueError(
            f"Cannot resume thread {actual_thread_id!r}: this session is "
            f"running thread {thread_id!r}."
        )
    result = graph.stream(
        Command(resume={"decisions": decisions}), config=config, stream_mode="updates"
    )
    return cast(Iterator[dict[str, Any]], result)


def stream_turn(
    graph: Any,
    payload: dict[str, Any],
    config: RunnableConfig,
    *,
    thread_id: str,
) -> None:
    """Stream one full turn, resolving every interrupt it raises along the way."""
    chunks = graph.stream(payload, config=config, stream_mode="updates")
    _drive(graph, chunks, config, thread_id=thread_id)


def _drive(
    graph: Any,
    chunks: Iterator[dict[str, Any]],
    config: RunnableConfig,
    *,
    thread_id: str,
) -> None:
    for chunk in chunks:
        if "__interrupt__" in chunk:
            interrupts = cast("tuple[Interrupt, ...]", chunk["__interrupt__"])
            decisions = _resolve_interrupt(interrupts)
            resumed = resume_turn(graph, config, decisions, thread_id=thread_id)
            _drive(graph, resumed, config, thread_id=thread_id)
            return
        _print_update(chunk)


def _resolve_interrupt(interrupts: tuple[Interrupt, ...]) -> list[Decision]:
    print(f"\n{_APPROVAL_BANNER}\nACTION REQUIRES APPROVAL\n{_APPROVAL_BANNER}")
    print(hitl.render_interrupt(interrupts[0].value))
    choice = _prompt_choice()
    text = ""
    if choice == "edit":
        text = _prompt_text("Your feedback: ")
    elif choice == "reject":
        text = _prompt_text("Reason: ")
    return hitl.build_decisions(choice, text)


def _prompt_choice() -> hitl.ReplChoice:
    valid_choices = ("approve", "edit", "reject")
    while True:
        raw = input("approve / edit / reject: ").strip().lower()
        if raw in valid_choices:
            return cast(hitl.ReplChoice, raw)
        print(f"Unrecognised choice {raw!r} -- type approve, edit or reject.")


def _prompt_text(prompt: str) -> str:
    return input(prompt).strip()


def _print_update(chunk: dict[str, Any]) -> None:
    for node_name, update in chunk.items():
        if node_name.endswith(".after_model") or not isinstance(update, dict):
            continue
        for message in update.get("messages", []):
            _print_message(message)


def _print_message(message: BaseMessage) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(format_tool_call(call["name"], call["args"]))
        text = message.text
        if text:
            print(f"Agent: {text}")
    elif isinstance(message, ToolMessage):
        print(format_tool_result(str(message.content)))


def _warm_retriever() -> None:
    try:
        get_retriever()
        print("Knowledge base retriever ready.")
    except (FileNotFoundError, IndexMismatchError) as error:
        print(f"Knowledge base retriever unavailable: {error}")


_EXIT_COMMANDS = frozenset({"exit", "quit"})


def _is_exit_command(text: str) -> bool:
    """Whether `text` is a whole-word request to end the REPL, not a
    research question that happens to mention "exit" or "quit"."""
    return text.strip().lower() in _EXIT_COMMANDS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse this process's command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse instead of `sys.argv[1:]`. Tests pass this
        explicitly instead of patching `sys.argv`.

    Returns
    -------
    argparse.Namespace
        `orchestration` is `None` when the flag was not given -- the signal
        to leave `Settings.orchestration` (env-configured, default
        `"supervisor"`) untouched, not a third orchestration value.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orchestration", choices=["supervisor", "graph"], default=None
    )
    return parser.parse_args(argv)


def _build_graph(settings: Settings) -> Any:
    """Build the compiled graph for `settings.orchestration`.

    Both `create_supervisor` and `create_orchestrator` return a graph the
    REPL drives identically through `stream_turn`/`resume_turn` -- this is
    the one place that chooses which of the two gets built.
    """
    if settings.orchestration == "supervisor":
        return create_supervisor(settings)
    return create_orchestrator(settings)


def main() -> None:
    configure_console()
    args = _parse_args()
    settings = load_settings()
    if args.orchestration is not None:
        settings = apply_overrides(settings, {"orchestration": args.orchestration})
    _warm_retriever()

    graph = _build_graph(settings)
    thread_id = str(uuid.uuid4())
    config = build_run_config(thread_id, settings)

    print("Ready. Ask a research question (or type exit/quit).")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if _is_exit_command(question):
            break
        stream_turn(
            graph, {"messages": [("user", question)]}, config, thread_id=thread_id
        )


if __name__ == "__main__":
    main()
