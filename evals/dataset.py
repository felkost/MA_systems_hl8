"""The end-to-end evaluation dataset: one row per full Plan -> Research ->
Critique -> save_report run, driven through `evals/run_eval.py`'s
`make_end_to_end_target`.

Eighteen rows: twelve questions the ingested corpus (`data/`) and the web can
answer between them, plus six rows written specifically to exercise a
multi-agent behaviour a single-agent Research Agent has no equivalent for --
a question whose natural answer risks being stale (freshness check), one
too vague to decompose without the Planner doing real work, and one
entirely outside the corpus (does the system say so, or fabricate a plan
around a request there is nothing to research).

Every `reference_outputs` field is optional, read by exactly one evaluator
in `evals/evaluators.py`; a row missing a field simply makes that evaluator
return `None` for it (never `0`) rather than requiring every row to answer
every evaluator's question -- `evals/evaluators.py::test_none_is_not_folded_into_zero`
is what makes that safe.
"""

from __future__ import annotations

from typing import Any, TypedDict

EvalReferenceOutputs = TypedDict(
    "EvalReferenceOutputs",
    {
        "expected_verdict": str | None,
        "is_out_of_scope": bool,
    },
    total=False,
)

EvalExample = TypedDict(
    "EvalExample",
    {"inputs": dict[str, str], "outputs": EvalReferenceOutputs},
)


def _row(
    question: str,
    *,
    expected_verdict: str | None = None,
    is_out_of_scope: bool = False,
) -> EvalExample:
    outputs: EvalReferenceOutputs = {}
    if expected_verdict is not None:
        outputs["expected_verdict"] = expected_verdict
    if is_out_of_scope:
        outputs["is_out_of_scope"] = True
    return {"inputs": {"question": question}, "outputs": outputs}


# The assignment's own worked example (docs/task-hl8.md) -- kept as row one
# so a spot-check against the console transcript in the assignment text is
# always one row away.
EVAL_EXAMPLES: list[EvalExample] = [
    _row(
        "Compare RAG approaches: naive, sentence-window, and parent-child. "
        "Write a report."
    ),
    _row(
        "What is retrieval-augmented generation and how does it reduce hallucination?"
    ),
    _row(
        "What is LangChain and what problem does it solve for LLM application "
        "developers?"
    ),
    _row(
        "What is a large language model and how does the transformer architecture "
        "enable it?"
    ),
    _row(
        "Summarize the architecture patterns used to coordinate multiple LLM agents, "
        "and name their tradeoffs."
    ),
    _row(
        "What is the Agent-as-a-Judge evaluation methodology, and how does it differ "
        "from single-response LLM-as-a-judge?"
    ),
    _row(
        "How does human-LLM collaborative planning work, and what role does the "
        "human play?"
    ),
    _row(
        "Explain the evaluator-optimizer workflow pattern and give an example of when "
        "to use it instead of a single agent."
    ),
    _row(
        "What are common failure modes of multi-agent LLM systems, and how can they be "
        "mitigated at the architecture level?"
    ),
    _row(
        "Explain human-in-the-loop middleware and how it gates a write tool "
        "before it runs."
    ),
    _row(
        "What are the risks of prompt injection for an agent that reads web pages with "
        "read_url, and how can a system contain them?"
    ),
    _row(
        "Compare the supervisor (agent-as-tool) and explicit StateGraph coordination "
        "patterns for a multi-agent system, and recommend when to use each."
    ),
    # Deliberately stale-fact: a natural answer risks citing old benchmark
    # numbers unless the Researcher/Critic actively check freshness.
    _row(
        "What are the most recent (2025-2026) benchmark results comparing RAG "
        "retrieval strategies?",
        expected_verdict="REVISE",
    ),
    # Under-specified: no scope, no format -- tests whether the Planner
    # decomposes it into concrete search queries instead of passing it
    # through unchanged.
    _row("Tell me about RAG."),
    # Out of scope: unrelated to the ingested corpus or to LLM/agent
    # research at all.
    _row("What's a good recipe for a Ukrainian borscht?", is_out_of_scope=True),
    _row(
        "What does the Agent-as-a-Judge paper say about trajectory-level evaluation, "
        "and how would it grade a multi-step agent run?"
    ),
    _row(
        "What risks does the multi-agent architecture survey identify, and how does "
        "human-in-the-loop approval mitigate one of them?"
    ),
    _row(
        "A researcher claims sentence-window retrieval always outperforms naive RAG. "
        "Verify this claim against the ingested sources and the web."
    ),
]


def as_langsmith_examples() -> list[dict[str, Any]]:
    """Render `EVAL_EXAMPLES` as the dict shape `Client.create_examples` expects."""
    return [
        {"inputs": row["inputs"], "outputs": row["outputs"]} for row in EVAL_EXAMPLES
    ]
