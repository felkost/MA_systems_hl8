"""Configuration read from the environment and `.env`.

Every tunable value lives here. Prompt text lives in `prompts.py` because
prompt version is an evaluation axis; this module ends with a re-export of
its four `build_*_prompt` functions, so a reviewer grepping `config.py`
finds all four names.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration for one process, shared by all four agents.

    Notes
    -----
    Only `api_key` and `model_name` need a `validation_alias`. Every other
    field is matched to its upper-case environment variable by name.
    """

    api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4.1-mini", validation_alias="MODEL_NAME")
    # Per-agent overrides for the "one model vs two" experiment (stage 11).
    # Unset means "use model_name" -- resolved by each agent factory, not
    # here, so an experiment can vary a single agent without a second field
    # per agent it did not touch.
    planner_model_name: str | None = None
    critic_model_name: str | None = None

    # Which entry each agent's own prompt registry runs on (prompts.py,
    # stage 2). An unregistered name is a hard error raised by the
    # corresponding build_*_prompt function, never a silent fallback.
    planner_prompt_version: str = "p1"
    researcher_prompt_version: str = "r1"
    critic_prompt_version: str = "c1"
    supervisor_prompt_version: str = "s1"

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    max_search_results: int = Field(default=5, ge=1, le=10)
    max_search_snippet_length: int = Field(default=500, ge=100, le=2000)
    max_url_content_length: int = Field(default=5000, ge=1000, le=10000)
    http_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)

    # recursion_limit is a graph-level backstop that only fires if a
    # tool-call limit somehow does not; the tool-call limits themselves are
    # the per-agent fields below, enforced by ToolCallLimitMiddleware.
    recursion_limit: int = Field(default=100, ge=2, le=200)

    # The Planner's own budget (4) is a small, fixed reconnaissance limit,
    # not a tunable the experiments vary, so it is not exposed as a setting.
    researcher_max_tool_calls: int = Field(default=8, ge=1, le=50)
    critic_max_tool_calls: int = Field(default=5, ge=1, le=50)
    # How many revision rounds the Critic may send the Researcher back for.
    # Both orchestration paths enforce this as a hard cap -- ToolCallLimitMiddleware
    # on the supervisor path, revision_round state arithmetic on the graph
    # path -- so the upper bound is a Field constraint, not a runtime check:
    # an impossible sweep cell must fail apply_overrides before it spends a
    # token, not after.
    max_revisions: int = Field(default=2, ge=1, le=3)

    # supervisor (agent-as-tool, as the assignment prescribes) or graph (the
    # explicit StateGraph orchestrator). Same public interface either way --
    # see main.py --orchestration.
    orchestration: Literal["supervisor", "graph"] = "supervisor"

    # Pages read_url may open before the next search, enforced by
    # ReadUrlCapMiddleware on the Researcher. None removes the cap.
    max_read_url_per_search: int | None = Field(default=2, ge=1, le=10)

    output_dir: str = "output"

    embedding_model: str = "text-embedding-3-small"
    # Truncate each embedding vector to this many dimensions. None keeps the
    # model's full width. The value is recorded in the index manifest, because
    # a retriever that reads an index built with a different width returns
    # wrong results instead of an error.
    embedding_dimensions: int | None = Field(default=None, ge=64, le=3072)
    data_dir: str = "data"
    index_dir: str = "index"
    collection_name: str = "knowledge_base"
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)

    reranker_model: str = "BAAI/bge-reranker-base"
    # "auto" selects cuda when torch reports a device, cpu otherwise. A
    # Literal, not a str: an unknown value is rejected at startup instead of
    # reaching torch or falling back to the CPU without a warning.
    reranker_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Candidates per ensemble arm, not in total. The reranker only reorders
    # what it receives, so documents missed here cannot be recovered.
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=3, ge=1, le=20)
    ensemble_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    model_cache_dir: str = ".cache/models"

    # Below this cross-encoder score the top match is still returned, but
    # flagged. A starting value, not a calibrated one: the reranker's real
    # score distribution on this corpus has not been measured.
    rerank_confidence_floor: float = 0.3
    max_knowledge_search_length: int = Field(default=3000, ge=500, le=10000)

    # The knowledge graph is optional: an unset password must not stop any
    # agent from starting, only stop graph_search and `ingest.py --graph`
    # from doing anything, the same way a missing index only disables
    # knowledge_search. docker-compose.yml requires NEO4J_PASSWORD to start
    # the container; Settings does not require it to start an agent.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"
    max_graph_search_length: int = Field(default=2000, ge=500, le=10000)

    # Read by both evals/upload_dataset.py and evals/run_eval.py, so the two
    # always agree on which LangSmith dataset holds the eval examples.
    eval_dataset_name: str = "research-agent-hl8-eval"
    # Separate dataset for the planner-only / critic-only rows: mixing them
    # into the end-to-end dataset would let a sub-agent experiment silently
    # pick up rows shaped for the full trajectory.
    subagent_eval_dataset_name: str = "research-agent-hl8-subagents"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "research-agent-hl8"
    langsmith_workspace_id: str | None = None

    # populate_by_name lets `model_validate` accept field names next to the
    # two environment aliases, which is what makes `apply_overrides` able to
    # re-validate a dumped instance instead of bypassing validation.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def _overlap_fits_in_a_chunk(self) -> Settings:
        """Reject an overlap that is too large for the chunk size.

        `RecursiveCharacterTextSplitter` raises this error only once it starts
        splitting, which happens after the documents are loaded. Both values
        are already known at startup.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _reranker_has_enough_candidates(self) -> Settings:
        """Reject a `rerank_top_n` larger than the candidate pool can hold.

        Each arm returns `retrieval_top_k` documents and the two arms overlap,
        so the merged pool holds between `retrieval_top_k` and twice that
        number. Only above the upper bound can the reranker never filter
        anything. A value between the two bounds is valid on a small corpus.
        """
        pool = 2 * self.retrieval_top_k
        if self.rerank_top_n > pool:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) exceeds the largest "
                f"possible candidate pool ({pool} = 2 x retrieval_top_k "
                f"{self.retrieval_top_k})"
            )
        return self


_output_dir_override: ContextVar[str | None] = ContextVar(
    "_output_dir_override", default=None
)


@contextmanager
def override_output_dir(path: str) -> Iterator[None]:
    """Redirect every `load_settings()` call in this context to `path`.

    A concurrent LangSmith `evaluate()` run calls `save_report` for several
    dataset examples at once, and `save_report` reads `Settings.output_dir`
    fresh on every call. A lock around it would serialise the whole
    experiment instead of isolating one example from another; a `ContextVar`
    does not, because each example wraps its run in `copy_context().run(...)`.
    """
    token = _output_dir_override.set(path)
    try:
        yield
    finally:
        _output_dir_override.reset(token)


def apply_overrides(settings: Settings, overrides: Mapping[str, object]) -> Settings:
    """Return a copy of `settings` with `overrides` applied and re-validated.

    Parameters
    ----------
    settings : Settings
        The configuration to start from.
    overrides : mapping of str to object
        Field names and their replacement values. An empty mapping returns
        `settings` unchanged.

    Returns
    -------
    Settings
        A new instance; `settings` itself is not modified.

    Raises
    ------
    pydantic.ValidationError
        If the result violates a field constraint or a cross-field rule.

    Notes
    -----
    `model_copy(update=...)` would be shorter and skips validation entirely.
    A measurement sweep sets these values from a table of candidate
    configurations, so an impossible one -- a `max_revisions` past its bound
    -- has to be refused before it spends real tokens, not after.
    """
    if not overrides:
        return settings
    return Settings.model_validate({**settings.model_dump(), **overrides})


def load_settings() -> Settings:
    """Build `Settings` from the environment and `.env`.

    Returns
    -------
    Settings
        Validated configuration. `output_dir` is replaced with the active
        `override_output_dir` value, if any.

    Raises
    ------
    pydantic.ValidationError
        If a required value is missing or out of range.

    Notes
    -----
    `model_validate({})` is used instead of `Settings()`. The values come from
    the environment, not from arguments, and the constructor form makes mypy
    report the required `api_key` as a missing argument.
    """
    settings = Settings.model_validate({})
    override = _output_dir_override.get()
    if override is None:
        return settings
    return settings.model_copy(update={"output_dir": override})


# Re-exported through module __getattr__ (PEP 562), not a static import.
# `tools`, `graph` and `retriever` each do `from config import Settings,
# load_settings` at module level; a static `from prompts import ...` here
# imports `prompts` eagerly to satisfy the re-export, and `prompts` imports
# `tools` to list the Planner's tools -- closing a cycle among these
# modules. Placing the import after `Settings`/`load_settings` are already
# defined only breaks that cycle when `config` happens to be the first of
# them to be imported, which is not guaranteed (e.g. `python -c "import
# tools"` hits it starting from the other side). Deferring the lookup to
# first attribute access removes the cycle instead of depending on import
# order to avoid it.
_PROMPT_BUILDER_NAMES = (
    "build_planner_prompt",
    "build_researcher_prompt",
    "build_critic_prompt",
    "build_supervisor_prompt",
)


def __getattr__(name: str) -> Any:
    if name in _PROMPT_BUILDER_NAMES:
        import prompts

        return getattr(prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Settings",
    "apply_overrides",
    "load_settings",
    "override_output_dir",
    *_PROMPT_BUILDER_NAMES,
]
