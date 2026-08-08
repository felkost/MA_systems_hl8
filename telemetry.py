"""Local telemetry: two JSONL streams recorded through one callback handler.

`output/logs/<session>-tools.jsonl` records one line per tool call, tagged
with which of the four agents (planner/researcher/critic/supervisor, or
orchestrator on the graph path) made it. `output/logs/<session>-retrieval.jsonl`
records one line per `knowledge_search` call: which documents each retrieval
arm returned, how many overlapped, and how the reranker moved every document
that survived into the final answer.

One handler, not two, because LangChain routes tool calls and retriever calls
through the same callback system, and `parent_run_id` is what lets the
retrieval stream be reconstructed without touching `retriever.py`. Confirmed
by reading the installed source of `EnsembleRetriever.rank_fusion`: each arm
is invoked as a child of the ensemble's own run, tagged `retriever_1`
(lexical, first in `build_retriever`'s `retrievers=[lexical, semantic]`) and
`retriever_2` (semantic). The ensemble itself is invoked as the
`ContextualCompressionRetriever`'s `base_retriever`, so it in turn is a child
of the outer run that `get_retriever().invoke()` returns -- the one call a
tool makes without touching this module at all, because `BaseTool.run` sets
the callback context before running the tool body.

The `agent` field on every tool event comes off `on_tool_start`'s own
`metadata` parameter, populated by each `create_*_agent`/`create_supervisor`/
`create_orchestrator` factory calling `.with_config(metadata={"agent": ...})`
on its returned compiled graph. A probe against the installed `langchain`/
`langgraph` confirmed this survives nested invocation -- a sub-agent's own
`agent` tag wins over an already-ambient one from whatever outer graph
invoked it, whether that outer caller is a `@tool` wrapper body (the
supervisor path) or a plain `StateGraph` node function body (the graph
path) -- so no per-call-site wiring is needed beyond the five factories.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

import paths
from config import Settings

LEXICAL_TAG = "retriever_1"
SEMANTIC_TAG = "retriever_2"

UNKNOWN_AGENT = "unknown"


def tools_log_path(settings: Settings, session_id: str) -> Path:
    """Locate the JSONL stream one tool call is appended to."""
    return _log_dir(settings) / f"{session_id}-tools.jsonl"


def retrieval_log_path(settings: Settings, session_id: str) -> Path:
    """Locate the JSONL stream one `knowledge_search` call is appended to."""
    return _log_dir(settings) / f"{session_id}-retrieval.jsonl"


def verdict_log_path(settings: Settings, session_id: str) -> Path:
    """Locate the JSONL stream one Critic verdict is appended to."""
    return _log_dir(settings) / f"{session_id}-verdicts.jsonl"


def log_verdict(
    settings: Settings,
    session_id: str,
    *,
    path: str,
    verdict: str,
    round: int,
) -> None:
    """Record one Critic verdict.

    Neither orchestration path exposes a verdict as a tool call --
    `supervisor.py`'s `critique` wrapper and `orchestrator.py`'s
    `critique_node` call this directly instead, since a verdict is a
    sub-agent's structured result, not a callback event `TelemetryHandler`
    can see.

    Parameters
    ----------
    settings : Settings
        Supplies `output_dir`.
    session_id : str
        The session this verdict belongs to -- the REPL's own `thread_id`,
        reused rather than minted separately.
    path : str
        Which coordination path produced the verdict: "supervisor" or
        "orchestrator".
    verdict : str
        "APPROVE" or "REVISE".
    round : int
        The revision round this verdict was reached in, 0-indexed.
    """
    _write_json_line(
        verdict_log_path(settings, session_id),
        {
            "ts": _now(),
            "session": session_id,
            "path": path,
            "verdict": verdict,
            "round": round,
        },
    )


def _log_dir(settings: Settings) -> Path:
    return paths.resolve(settings.output_dir) / "logs"


def _document_id(document: Document) -> str:
    """Identify a chunk across retrieval arms.

    Not `ingest._chunk_id`: that one also hashes the chunk text, because it
    has to detect an edit that keeps the same offsets. Here the id only needs
    to be stable across the arms of a single query, and `source`/`page`/
    `start_index` -- metadata every arm already carries -- is enough for that.
    """
    metadata = document.metadata
    return (
        f"{metadata.get('source')}|{metadata.get('page')}|{metadata.get('start_index')}"
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe_score(score: Any) -> float | None:
    """Cast `rerank_score` to a plain `float`.

    `ScoredCrossEncoderReranker` writes the cross-encoder's own numpy/torch
    `float32`, which `json.dumps` rejects -- and LangChain's callback manager
    swallows the resulting exception silently, so a real `knowledge_search`
    call produced no `retrieval.jsonl` row at all rather than a visible error.
    """
    return None if score is None else float(score)


def _write_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tool_output_text(output: Any) -> str:
    """Read the text a tool produced off whatever `on_tool_end` was handed.

    `BaseTool.run` passes a `ToolMessage` when the call came from an agent
    step (it carries `tool_call_id`), and a bare string when a tool is called
    directly, as some of this project's own tests do.
    """
    content = getattr(output, "content", output)
    return content if isinstance(content, str) else str(content)


@dataclass
class _ToolCall:
    name: str
    args: dict[str, Any]
    agent: str
    started: float


@dataclass
class _RetrieverRun:
    parent_run_id: UUID | None
    tag: str | None
    query: str
    started: float
    documents: list[Document] = field(default_factory=list)


class TelemetryHandler(BaseCallbackHandler):
    """Record tool calls and retrieval internals to two JSONL streams.

    A callback rather than wrappers around each tool: it sees every call any
    agent makes through any path, nests correctly inside the LangSmith trace
    instead of competing with it, and -- because sub-retrievers report their
    own runs -- it can reconstruct which arm produced which document.
    """

    def __init__(self, settings: Settings, session_id: str) -> None:
        self._session_id = session_id
        self._tools_path = tools_log_path(settings, session_id)
        self._retrieval_path = retrieval_log_path(settings, session_id)
        self._tool_calls: dict[UUID, _ToolCall] = {}
        self._retriever_runs: dict[UUID, _RetrieverRun] = {}
        self._retriever_children: dict[UUID, list[UUID]] = {}

    # -- tools ------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_calls[run_id] = _ToolCall(
            name=str(serialized.get("name", "unknown")),
            args=inputs or {},
            agent=str((metadata or {}).get("agent", UNKNOWN_AGENT)),
            started=time.perf_counter(),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call = self._tool_calls.pop(run_id, None)
        text = _tool_output_text(output)
        self._write_tool_event(
            call=call,
            run_id=run_id,
            ok=not text.startswith("ERROR:"),
            result_chars=len(text),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        call = self._tool_calls.pop(run_id, None)
        self._write_tool_event(call=call, run_id=run_id, ok=False, result_chars=0)

    def _write_tool_event(
        self,
        call: _ToolCall | None,
        run_id: UUID,
        ok: bool,
        result_chars: int,
    ) -> None:
        name = call.name if call else "unknown"
        args = call.args if call else {}
        agent = call.agent if call else UNKNOWN_AGENT
        seconds = round(time.perf_counter() - call.started, 4) if call else 0.0
        _write_json_line(
            self._tools_path,
            {
                "ts": _now(),
                "session": self._session_id,
                "tool": name,
                "agent": agent,
                "args": args,
                "ok": ok,
                "seconds": seconds,
                "result_chars": result_chars,
                "run_id": str(run_id),
            },
        )

    # -- retrievers ---------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tag = next((t for t in tags or [] if t in (LEXICAL_TAG, SEMANTIC_TAG)), None)
        self._retriever_runs[run_id] = _RetrieverRun(
            parent_run_id=parent_run_id,
            tag=tag,
            query=query,
            started=time.perf_counter(),
        )
        if parent_run_id is not None:
            self._retriever_children.setdefault(parent_run_id, []).append(run_id)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._retriever_runs.get(run_id)
        if run is None:
            return
        run.documents = list(documents)

        # The outer run of one `knowledge_search` call is the only one whose
        # parent is not itself a retriever run this handler is tracking.
        if run.parent_run_id is None or run.parent_run_id not in self._retriever_runs:
            self._write_retrieval_event(run_id, run)

    def _write_retrieval_event(self, run_id: UUID, outer: _RetrieverRun) -> None:
        seconds = round(time.perf_counter() - outer.started, 4)
        ensemble_id, ensemble = self._single_child(run_id)
        ensemble_ids = [_document_id(d) for d in ensemble.documents] if ensemble else []

        lexical = semantic = None
        if ensemble_id is not None:
            lexical = self._tagged_child(ensemble_id, LEXICAL_TAG)
            semantic = self._tagged_child(ensemble_id, SEMANTIC_TAG)
        bm25_ids = [_document_id(d) for d in lexical.documents] if lexical else []
        semantic_ids = [_document_id(d) for d in semantic.documents] if semantic else []

        reranked = []
        for position, document in enumerate(outer.documents, start=1):
            document_id = _document_id(document)
            rank_before = (
                ensemble_ids.index(document_id) + 1
                if document_id in ensemble_ids
                else None
            )
            reranked.append(
                {
                    "id": document_id,
                    "rank_before": rank_before,
                    "rank_after": position,
                    "score": _json_safe_score(document.metadata.get("rerank_score")),
                }
            )

        _write_json_line(
            self._retrieval_path,
            {
                "ts": _now(),
                "session": self._session_id,
                "query": outer.query,
                "semantic_ids": semantic_ids,
                "bm25_ids": bm25_ids,
                "overlap": len(set(semantic_ids) & set(bm25_ids)),
                "ensemble_ids": ensemble_ids,
                "reranked": reranked,
                "dropped": max(len(ensemble_ids) - len(outer.documents), 0),
                "seconds": seconds,
            },
        )
        self._forget(run_id)

    def _single_child(self, run_id: UUID) -> tuple[UUID | None, _RetrieverRun | None]:
        children = self._retriever_children.get(run_id, [])
        if not children:
            return None, None
        child_id = children[0]
        return child_id, self._retriever_runs.get(child_id)

    def _tagged_child(self, parent_id: UUID, tag: str) -> _RetrieverRun | None:
        for child_id in self._retriever_children.get(parent_id, []):
            child = self._retriever_runs.get(child_id)
            if child is not None and child.tag == tag:
                return child
        return None

    def _forget(self, run_id: UUID) -> None:
        for child_id in self._retriever_children.pop(run_id, []):
            self._forget(child_id)
        self._retriever_runs.pop(run_id, None)
