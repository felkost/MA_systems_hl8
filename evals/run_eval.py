"""Eval runners: drives the Supervisor/Orchestrator (or a lone sub-agent)
through a LangSmith dataset and scores the result.

`evaluate()` needs a target callable that turns one dataset example's
`inputs` into `outputs` with no human in the loop -- an evaluation sweep
that hangs at the first `save_report` interrupt burns the whole session, so
`drive_to_completion` auto-approves every interrupt a run raises instead of
waiting for one. Each example also gets its own `thread_id` and its own
absolute `output_dir` (via `config.override_output_dir`), so concurrent
examples never share a checkpointer thread, a `save_report` filename
collision, or a telemetry stream -- `telemetry.py` resolves `output_dir`
through `paths.resolve`, anchored to `PROJECT_ROOT`, not the working
directory, so an isolation mechanism proven for `save_report` alone is not
evidence it isolates telemetry too (this bit a real run at stage 8).
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langsmith import Client

import telemetry
from agents.critic import create_critic_agent
from agents.planner import create_planner_agent
from config import Settings, load_settings, override_output_dir
from evals.evaluators import (
    critic_verified,
    hitl_respected,
    plan_covers_request,
    plan_is_structured,
    revision_converged,
    verdict_is_justified,
)
from evals import trajectory
from evals.trajectory import trajectory_match
from evals.upload_dataset import build_client
from orchestrator import create_orchestrator
from schemas import render_critique, render_plan
from supervisor import create_supervisor

APPROVE_DECISIONS: list[dict[str, str]] = [{"type": "approve"}]

PLANNER_EVALUATORS = [plan_is_structured, plan_covers_request]
CRITIC_EVALUATORS = [critic_verified, verdict_is_justified]


def end_to_end_evaluators(settings: Settings) -> list[Callable[..., dict[str, Any]]]:
    """Build the end-to-end evaluator list, bound to `settings`.

    A function, not a module-level constant: `trajectory_accuracy_judge`
    needs a real `ChatOpenAI` built from `settings.api_key`
    (`evals/trajectory.py::make_trajectory_accuracy_judge`), so the whole
    list can only be assembled once `settings` is known, not at import time.
    """
    return [
        plan_is_structured,
        critic_verified,
        verdict_is_justified,
        revision_converged,
        hitl_respected,
        trajectory_match,
        trajectory.make_trajectory_accuracy_judge(settings),
    ]


def drive_to_completion(
    graph: Any, payload: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Invoke a graph and auto-approve every HITL interrupt it raises.

    Parameters
    ----------
    graph : object with an `.invoke` method
        A compiled Supervisor or Orchestrator graph.
    payload : dict
        The first turn's input, e.g. `{"messages": [("user", question)]}`.
    config : dict
        Run config, carrying at least `configurable.thread_id`.

    Returns
    -------
    dict
        The final state, once no `__interrupt__` remains. A batch
        evaluation has no human to answer with `edit`/`reject`, so every
        pending decision is `approve`.
    """
    result = graph.invoke(payload, config=config)
    while "__interrupt__" in result:
        result = graph.invoke(
            Command(resume={"decisions": APPROVE_DECISIONS}), config=config
        )
    return result


def make_end_to_end_target(
    settings: Settings, graph: Any, *, output_root: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the `target` callable `evaluate()` calls once per example.

    Parameters
    ----------
    settings : Settings
        Base configuration; only `output_dir` is overridden, per example.
    graph : object with an `.invoke` method
        A compiled Supervisor or Orchestrator graph, built once and reused
        across every example -- safe because each call mints its own
        `thread_id`, and neither graph keeps mutable state outside its
        checkpointer.
    output_root : str
        Absolute directory each example gets its own subdirectory under.

    Returns
    -------
    callable
        `target(inputs) -> outputs`. `outputs` carries the final message
        list plus the paths to this example's own telemetry streams, so an
        evaluator can read the real trajectory instead of only what
        `outputs` itself exposes.
    """

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        thread_id = f"eval-{uuid.uuid4()}"
        example_output_dir = str(Path(output_root).resolve() / thread_id)
        run_settings = settings.model_copy(update={"output_dir": example_output_dir})
        with override_output_dir(example_output_dir):
            telemetry_handler = telemetry.TelemetryHandler(
                run_settings, session_id=thread_id
            )
            config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.recursion_limit,
                "callbacks": [telemetry_handler],
            }
            result = drive_to_completion(
                graph, {"messages": [("user", inputs["question"])]}, config
            )
        return {
            "messages": result.get("messages", []),
            "thread_id": thread_id,
            "output_dir": example_output_dir,
            "tools_log": str(telemetry.tools_log_path(run_settings, thread_id)),
            "verdicts_log": str(telemetry.verdict_log_path(run_settings, thread_id)),
            # `evaluators.revision_converged` reads this to tell a
            # deliberate cap-out (REVISE at the last allowed round) from a
            # run that simply never converged -- without it, that branch is
            # unreachable and the metric collapses to "last verdict ==
            # APPROVE" regardless of `max_revisions`, silently defeating the
            # one experiment (stage 11, #2) that varies this setting.
            "max_revisions": settings.max_revisions,
        }

    return target


def make_planner_target(
    settings: Settings,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the `target` callable for the planner-only sub-agent dataset."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = create_planner_agent(settings).invoke(
            {"messages": [HumanMessage(inputs["question"])]}
        )
        plan = result["structured_response"]
        return {"plan": plan.model_dump(), "rendered": render_plan(plan)}

    return target


def make_critic_target(
    settings: Settings,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the `target` callable for the critic-only sub-agent dataset."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = create_critic_agent(settings).invoke(
            {"messages": [HumanMessage(inputs["critique_input"])]}
        )
        critique = result["structured_response"]
        return {
            "critique": critique.model_dump(),
            "rendered": render_critique(critique),
        }

    return target


def run_evaluation(
    settings: Settings | None = None,
    *,
    orchestration: str | None = None,
    experiment_prefix: str | None = None,
    max_concurrency: int = 4,
    client: Client | None = None,
) -> Any:
    """Run the end-to-end dataset through one orchestration path.

    Parameters
    ----------
    settings : Settings, optional
        Loaded from the environment when omitted.
    orchestration : {"supervisor", "graph"}, optional
        Overrides `settings.orchestration` for this run.
    experiment_prefix : str, optional
        Defaults to `f"hl8-{orchestration}"`.
    max_concurrency : int, default 4
        Forwarded to `langsmith.evaluate` -- LangSmith's own
        `ContextThreadPoolExecutor` copies the calling context into every
        submitted example, which is what makes `override_output_dir`
        (a `ContextVar`) isolate concurrent examples correctly; confirmed
        against the installed `langsmith` before this was trusted for a
        real sweep.
    client : Client, optional
        Reused across calls when given; a fresh one is created otherwise.

    Returns
    -------
    langsmith.evaluation.ExperimentResults
        Typed `Any` here, deliberately: the concrete class lives in
        `langsmith.evaluation._runner`, a private module this project does
        not depend on directly.
    """
    settings = settings or load_settings()
    orchestration = orchestration or settings.orchestration
    graph = (
        create_supervisor(settings)
        if orchestration == "supervisor"
        else (create_orchestrator(settings))
    )
    output_root = str(Path(settings.output_dir).resolve() / "eval-runs")
    target = make_end_to_end_target(settings, graph, output_root=output_root)

    return (client or build_client(settings)).evaluate(
        target,
        data=settings.eval_dataset_name,
        evaluators=end_to_end_evaluators(settings),
        experiment_prefix=experiment_prefix or f"hl8-{orchestration}",
        max_concurrency=max_concurrency,
        metadata={"orchestration": orchestration},
    )


def run_subagent_evaluation(
    settings: Settings | None = None,
    *,
    experiment_prefix: str | None = None,
    max_concurrency: int = 4,
    client: Client | None = None,
) -> tuple[Any, Any]:
    """Run the planner-only and critic-only rows of the sub-agent dataset.

    Returns
    -------
    tuple of langsmith.evaluation.ExperimentResults
        `(planner_results, critic_results)`.
    """
    settings = settings or load_settings()
    client = client or build_client(settings)

    def _split(kind: str) -> Any:
        # `split`, not `metadata` -- LangSmith's native row-grouping field,
        # set on each row by `evals/upload_dataset.py` and read back here so
        # the planner-only and critic-only rows evaluate independently
        # despite sharing one dataset (`Settings.subagent_eval_dataset_name`).
        return client.list_examples(
            dataset_name=settings.subagent_eval_dataset_name, splits=[kind]
        )

    planner_results = client.evaluate(
        make_planner_target(settings),
        data=_split("planner"),
        evaluators=PLANNER_EVALUATORS,
        experiment_prefix=experiment_prefix or "hl8-planner",
        max_concurrency=max_concurrency,
    )
    critic_results = client.evaluate(
        make_critic_target(settings),
        data=_split("critic"),
        evaluators=CRITIC_EVALUATORS,
        experiment_prefix=experiment_prefix or "hl8-critic",
        max_concurrency=max_concurrency,
    )
    return planner_results, critic_results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orchestration", choices=["supervisor", "graph"], default=None
    )
    parser.add_argument(
        "--subagents",
        action="store_true",
        help="Run the sub-agent dataset instead of the end-to-end one",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    if args.subagents:
        run_subagent_evaluation(settings, max_concurrency=args.max_concurrency)
    else:
        run_evaluation(
            settings,
            orchestration=args.orchestration,
            max_concurrency=args.max_concurrency,
        )


if __name__ == "__main__":
    main()
