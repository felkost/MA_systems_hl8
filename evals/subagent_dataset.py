"""Sub-agent-only rows: the Planner and the Critic invoked directly, with no
Supervisor/Orchestrator around them and no `save_report` interrupt to drive
through. Isolates what each sub-agent contributes on its own, the way the "Planner
on/off" experiment and the Critic prompt `c1` vs `c2` comparison both need
real per-sub-agent numbers rather than only the numbers a full trajectory
produces.

Both kinds share one LangSmith dataset (`Settings.subagent_eval_dataset_name`)
rather than two, distinguished by the `split` field `evals/upload_dataset.py`
writes onto each row and `evals/run_eval.py::run_subagent_evaluation` reads
back with `Client.list_examples(..., splits=[...])` -- one dataset avoids the
two datasets silently drifting out of sync with `Settings`.
"""

from __future__ import annotations

from typing import TypedDict

from supervisor import CRITIQUE_INPUT_TEMPLATE

SubagentExample = TypedDict(
    "SubagentExample",
    {"inputs": dict[str, str], "outputs": dict[str, str], "split": str},
)


def _planner_row(question: str, *, expected_sources: str) -> SubagentExample:
    return {
        "inputs": {"question": question},
        "outputs": {"expected_sources": expected_sources},
        "split": "planner",
    }


def _critic_row(
    request: str, findings: str, *, expected_verdict: str
) -> SubagentExample:
    return {
        "inputs": {
            "critique_input": CRITIQUE_INPUT_TEMPLATE.format(
                request=request, findings=findings
            )
        },
        "outputs": {"expected_verdict": expected_verdict},
        "split": "critic",
    }


PLANNER_EXAMPLES: list[SubagentExample] = [
    _planner_row(
        "Compare naive, sentence-window and parent-child RAG.",
        expected_sources="both",
    ),
    _planner_row(
        "What is retrieval-augmented generation?", expected_sources="knowledge_base"
    ),
    _planner_row(
        "What are the newest (2026) LLM agent evaluation benchmarks?",
        expected_sources="web",
    ),
    _planner_row("Tell me about RAG.", expected_sources="knowledge_base"),
    _planner_row(
        "Summarize the failure modes multi-agent architecture surveys describe.",
        expected_sources="knowledge_base",
    ),
]

_COMPLETE_FRESH_FINDINGS = (
    "Naive RAG retrieves top-k chunks by embedding similarity alone. "
    "Sentence-window retrieval expands each retrieved sentence with its "
    "surrounding context before generation. Parent-child chunking indexes "
    "small child chunks for search but returns the larger parent chunk to "
    "the model. Source: retrieval-augmented-generation.pdf, and a 2026 "
    "benchmark article comparing all three on retrieval precision "
    "(https://example.com/rag-benchmarks-2026)."
)
_NO_CITATIONS_FINDINGS = (
    "RAG systems generally work by retrieving relevant text and feeding it "
    "to a language model. Sentence-window and parent-child are both "
    "variants that try to give the model more context than naive retrieval. "
    "This is based on general knowledge of the field."
)
_STALE_FINDINGS = (
    "According to a 2022 survey, naive RAG remains the dominant retrieval "
    "strategy in production systems, with sentence-window and parent-child "
    "still considered experimental. Source: a 2022 blog post."
)

CRITIC_EXAMPLES: list[SubagentExample] = [
    _critic_row(
        "Compare naive, sentence-window and parent-child RAG.",
        _COMPLETE_FRESH_FINDINGS,
        expected_verdict="APPROVE",
    ),
    _critic_row(
        "Compare naive, sentence-window and parent-child RAG.",
        _NO_CITATIONS_FINDINGS,
        expected_verdict="REVISE",
    ),
    _critic_row(
        "What are the newest (2026) benchmark results for RAG retrieval?",
        _STALE_FINDINGS,
        expected_verdict="REVISE",
    ),
    _critic_row(
        "What is a large language model?",
        "A large language model is a neural network, typically a "
        "transformer, trained on large text corpora to predict the next "
        "token; this enables in-context learning and few-shot task "
        "performance. Source: large-language-model.pdf.",
        expected_verdict="APPROVE",
    ),
]
