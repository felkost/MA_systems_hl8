"""Uploads `evals/dataset.py` and `evals/subagent_dataset.py` to their two
LangSmith datasets (`Settings.eval_dataset_name` /
`Settings.subagent_eval_dataset_name`), creating each dataset the first time
it is run and leaving an already-created one alone.

Additive, not idempotent on row content: running this twice against a
dataset that already has rows uploads a second copy of each. This project
runs it by hand, occasionally, rather than from CI -- `_ensure_dataset`'s
existence check is what stops a second run from duplicating the *dataset*,
and re-running after actually editing `evals/dataset.py` is expected to mean
clearing the dataset in the LangSmith UI first, not appending on top of it.
"""

from __future__ import annotations

from typing import Any, Protocol

from config import Settings, load_settings
from evals.dataset import EVAL_EXAMPLES
from evals.subagent_dataset import CRITIC_EXAMPLES, PLANNER_EXAMPLES


class DatasetClient(Protocol):
    """The subset of `langsmith.Client` this module calls.

    A `Protocol`, not `langsmith.Client` itself, so a test double needs to
    implement only these three methods instead of the real client's full
    surface.
    """

    def has_dataset(self, *, dataset_name: str) -> bool: ...

    def create_dataset(self, name: str, **kwargs: Any) -> Any: ...

    def create_examples(
        self, *, dataset_name: str, examples: list[dict[str, Any]]
    ) -> Any: ...


def _ensure_dataset(client: DatasetClient, name: str, description: str) -> None:
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description=description)


def upload_eval_dataset(
    settings: Settings | None = None, *, client: DatasetClient | None = None
) -> None:
    """Create (if missing) and populate the end-to-end evaluation dataset."""
    settings = settings or load_settings()
    client = client or build_client(settings)
    _ensure_dataset(
        client,
        settings.eval_dataset_name,
        "End-to-end Plan -> Research -> Critique -> save_report runs, "
        "one row per user question.",
    )
    client.create_examples(
        dataset_name=settings.eval_dataset_name,
        examples=[
            {"inputs": row["inputs"], "outputs": row["outputs"]}
            for row in EVAL_EXAMPLES
        ],
    )


def upload_subagent_dataset(
    settings: Settings | None = None, *, client: DatasetClient | None = None
) -> None:
    """Create (if missing) and populate the sub-agent-only dataset.

    Planner-only and critic-only rows share one dataset, distinguished by
    the `split` field each row carries -- `evals/run_eval.py` reads it back
    with `Client.list_examples(..., splits=[...])`.
    """
    settings = settings or load_settings()
    client = client or build_client(settings)
    _ensure_dataset(
        client,
        settings.subagent_eval_dataset_name,
        "Planner-only and critic-only rows, invoked directly with no "
        "Supervisor/Orchestrator around them.",
    )
    client.create_examples(
        dataset_name=settings.subagent_eval_dataset_name,
        examples=[
            {"inputs": row["inputs"], "outputs": row["outputs"], "split": row["split"]}
            for row in (*PLANNER_EXAMPLES, *CRITIC_EXAMPLES)
        ],
    )


def build_client(settings: Settings) -> Any:
    """A real `langsmith.Client`, built explicitly from `Settings`.

    Shared with `evals/run_eval.py`, which needs the identical credential
    handling for its own default `Client()` -- unlike the deliberate
    per-orchestration-path duplication elsewhere in this project (e.g.
    `_build_model` in each agent factory), there is only one correct way to
    authenticate to LangSmith, so fixing it once here is what keeps a future
    fix from being needed twice.

    `Client()` with no arguments reads `LANGSMITH_API_KEY` from
    `os.environ`, which pydantic-settings never populates from `.env` on its
    own -- confirmed by a real run that failed with `LangSmithAuthError:
    ... "Invalid token"` before this fix, even though `.env` held a valid
    key. `tracing.configure_tracing` is *not* the right fix here, even
    though it exists for exactly this os.environ-vs-`.env` gap: it clears
    `LANGSMITH_API_KEY` whenever `settings.langsmith_tracing` is `False`,
    conflating "no trace spans wanted" with "no LangSmith access at all" --
    the two are independent for this module, whose whole job is talking to
    the LangSmith evaluation API regardless of the tracing toggle. Passing
    `api_key`/`api_url` explicitly sidesteps both the ambient environment
    and that toggle.

    Typed `Any`: `Client.create_dataset`'s first parameter is named
    `dataset_name`, not `name`, so the real class fails `DatasetClient`'s
    structural check on parameter names alone even though every call site
    here passes it positionally and works at runtime -- a mismatch in the
    `Protocol`'s shape, not a real defect.
    """
    from langsmith import Client

    if settings.langsmith_api_key is None:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. The evaluation harness needs a "
            "real LangSmith connection regardless of langsmith_tracing."
        )
    return Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
        workspace_id=settings.langsmith_workspace_id,
    )


def main() -> None:
    settings = load_settings()
    client = build_client(settings)
    upload_eval_dataset(settings, client=client)
    upload_subagent_dataset(settings, client=client)


if __name__ == "__main__":
    main()
