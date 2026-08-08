"""Structured outputs shared by the Planner, Critic and the report write path.

Field sets for `ResearchPlan` and `CritiqueResult` are copied verbatim from
`docs/task-hl8.md`, not designed from scratch: `sources_to_check` stays
`list[str]` rather than a `Literal`, because its own description allows the
model to answer "both", which a `Literal["knowledge_base", "web"]` would
reject.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# OpenAI's strict structured-output mode refuses any schema that does not
# supply `additionalProperties: false`, and a plain Pydantic model omits the
# key entirely. `extra="forbid"` is what emits it. Strict mode matters
# because without it the provider treats `required` as a hint: a
# `CritiqueResult` really did come back with no `verdict` at all.
_STRUCTURED_OUTPUT_CONFIG = ConfigDict(extra="forbid")


class ResearchPlan(BaseModel):
    """A decomposed research request, produced by the Planner."""

    model_config = _STRUCTURED_OUTPUT_CONFIG

    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(
        min_length=1, description="Specific queries to execute"
    )
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', or both")
    output_format: str = Field(description="What the final report should look like")


class CritiqueResult(BaseModel):
    """The Critic's verdict on one round of research findings."""

    model_config = _STRUCTURED_OUTPUT_CONFIG

    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(
        description="Is the data up-to-date and based on recent sources?"
    )
    is_complete: bool = Field(
        description="Does the research fully cover the user's original request?"
    )
    is_well_structured: bool = Field(
        description="Are findings logically organized and ready for a report?"
    )
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(
        description="What is missing, outdated, or poorly structured"
    )
    revision_requests: list[str] = Field(
        description="Specific things to fix if verdict is REVISE"
    )


class ReportDraft(BaseModel):
    """A report ready to hand to `save_report`."""

    filename: str = Field(description="Name for the saved report file")
    content: str = Field(description="Markdown content of the report")


def render_plan(plan: ResearchPlan) -> str:
    """Render a `ResearchPlan` as compact markdown for the Supervisor.

    Parameters
    ----------
    plan : ResearchPlan
        The Planner's structured response.

    Returns
    -------
    str
        Markdown text. The Supervisor reads this instead of
        `result["structured_response"]` directly, so it never sees raw JSON.
    """
    queries = "\n".join(f"- {query}" for query in plan.search_queries)
    sources = ", ".join(plan.sources_to_check)
    return (
        f"**Goal:** {plan.goal}\n\n"
        f"**Search queries:**\n{queries}\n\n"
        f"**Sources to check:** {sources}\n\n"
        f"**Output format:** {plan.output_format}"
    )


def render_critique(critique: CritiqueResult) -> str:
    """Render a `CritiqueResult` as compact markdown for the Supervisor.

    Parameters
    ----------
    critique : CritiqueResult
        The Critic's structured response.

    Returns
    -------
    str
        Markdown text, with empty `gaps`/`revision_requests` spelled out as
        "none" rather than left as an empty section.
    """
    strengths = _render_list(critique.strengths)
    gaps = _render_list(critique.gaps)
    revision_requests = _render_list(critique.revision_requests)
    return (
        f"**Verdict:** {critique.verdict}\n\n"
        f"**Fresh:** {critique.is_fresh} · "
        f"**Complete:** {critique.is_complete} · "
        f"**Well-structured:** {critique.is_well_structured}\n\n"
        f"**Strengths:**\n{strengths}\n\n"
        f"**Gaps:**\n{gaps}\n\n"
        f"**Revision requests:**\n{revision_requests}"
    )


def _render_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)
