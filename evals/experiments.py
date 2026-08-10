"""Stage-11 experiment matrix: `max_revisions` and the Critic prompt version.

Each cell overrides one `Settings` field via `apply_overrides` and calls
`run_eval.run_evaluation` with its own `experiment_prefix` -- the default
`f"hl8-{orchestration}"` is identical for every arm of a same-path
comparison, so a shared-dataset sweep run under it cannot be told apart
afterwards.

Experiment 4 (`c1` vs `c2`) runs on the graph path only: `supervisor.py`'s
`plan`/`research`/`critique` wrappers are module-level `@tool` functions
that call `load_settings()` internally, so a `critic_prompt_version`
override never reaches them on the supervisor path -- both arms would
measure one configuration twice (pre-flight finding, same file).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langsmith import Client

from config import Settings, apply_overrides, load_settings
from evals.run_eval import run_evaluation


@dataclass(frozen=True)
class ExperimentCell:
    """One matrix row: an orchestration path, its setting overrides, and
    the `experiment_prefix` that keeps it distinguishable in LangSmith."""

    experiment_prefix: str
    orchestration: str
    overrides: dict[str, Any]


def _experiment_2_matrix() -> list[ExperimentCell]:
    """`max_revisions` in {1, 2, 3}, both orchestration paths -- 6 cells."""
    return [
        ExperimentCell(
            experiment_prefix=f"hl8-exp2-{orchestration}-mr{max_revisions}",
            orchestration=orchestration,
            overrides={"max_revisions": max_revisions},
        )
        for orchestration in ("supervisor", "graph")
        for max_revisions in (1, 2, 3)
    ]


def _experiment_4_matrix(repeats: int = 3) -> list[ExperimentCell]:
    """`c1` vs `c2`, graph path only, `repeats` sweeps per arm.

    `verdict_is_justified` moved 0.78 -> 0.50/0.64 across two sweeps of
    identical code on the identical dataset, a spread of ~14 points at
    n=18 measured on 2026-08-09 -- one sweep per arm cannot tell the
    prompts apart.
    """
    return [
        ExperimentCell(
            experiment_prefix=f"hl8-exp4-graph-{version}-run{run}",
            orchestration="graph",
            overrides={"critic_prompt_version": version},
        )
        for version in ("c1", "c2")
        for run in range(1, repeats + 1)
    ]


EXPERIMENT_2_MATRIX = _experiment_2_matrix()
EXPERIMENT_4_MATRIX = _experiment_4_matrix()


def validate_matrix(base_settings: Settings, cells: list[ExperimentCell]) -> None:
    """Raise on the first cell `apply_overrides` rejects.

    Parameters
    ----------
    base_settings : Settings
        The configuration each cell's overrides apply on top of.
    cells : list of ExperimentCell
        The matrix to check.

    Raises
    ------
    pydantic.ValidationError
        If any cell's overrides produce an invalid `Settings`. Checking
        every cell before a single one runs is what keeps a broken cell
        from wasting the tokens already spent on the cells before it.
    """
    for cell in cells:
        apply_overrides(base_settings, cell.overrides)


def run_cell(
    cell: ExperimentCell,
    *,
    settings: Settings,
    client: Client | None = None,
) -> Any:
    """Run one matrix cell's sweep through `run_eval.run_evaluation`."""
    cell_settings = apply_overrides(settings, cell.overrides)
    return run_evaluation(
        cell_settings,
        orchestration=cell.orchestration,
        experiment_prefix=cell.experiment_prefix,
        client=client,
    )


def run_matrix(
    cells: list[ExperimentCell],
    *,
    settings: Settings | None = None,
    skip: Sequence[str] = (),
) -> dict[str, Any]:
    """Run every cell in `cells` sequentially, keyed by `experiment_prefix`.

    Parameters
    ----------
    cells : list of ExperimentCell
        The matrix to run.
    settings : Settings, optional
        Loaded from the environment when omitted.
    skip : sequence of str, optional
        `experiment_prefix` values to leave unrun -- cells that already
        landed in LangSmith on an earlier attempt. The whole matrix is
        still validated: a skipped cell being invalid is a broken matrix
        either way.

    Returns
    -------
    dict
        `experiment_prefix` -> that cell's results. Skipped cells are
        absent rather than present with a null.

    Notes
    -----
    Sequential across cells, not just within one: each cell is already a
    concurrent sweep over the dataset (`run_evaluation`'s own
    `max_concurrency`), and running several cells at once would spend real
    API budget with no chance to stop between them if one looks wrong.
    """
    base = settings or load_settings()
    validate_matrix(base, cells)
    skipped = set(skip)
    results: dict[str, Any] = {}
    for cell in cells:
        if cell.experiment_prefix in skipped:
            print(f"[{cell.experiment_prefix}] skipped, already run")
            continue
        print(
            f"[{cell.experiment_prefix}] orchestration={cell.orchestration} "
            f"overrides={cell.overrides}"
        )
        results[cell.experiment_prefix] = run_cell(cell, settings=base)
    return results


def find_cell(cells: list[ExperimentCell], experiment_prefix: str) -> ExperimentCell:
    """Look up one cell in a matrix by its `experiment_prefix`.

    Raises
    ------
    KeyError
        If no cell in `cells` carries that prefix.
    """
    for cell in cells:
        if cell.experiment_prefix == experiment_prefix:
            return cell
    raise KeyError(
        f"No cell with experiment_prefix {experiment_prefix!r}. "
        f"Known: {[c.experiment_prefix for c in cells]}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["2", "4"], required=True)
    parser.add_argument(
        "--only",
        default=None,
        metavar="PREFIX",
        help=(
            "run exactly this one cell (by experiment_prefix) and exit -- "
            "run one process per cell from a shell loop for a sweep that "
            "survives a crash in any single cell, instead of losing every "
            "cell still queued behind it in one long-lived process"
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="experiment_prefix values that already ran; do not spend on them again",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    matrix = EXPERIMENT_2_MATRIX if args.experiment == "2" else EXPERIMENT_4_MATRIX
    if args.only is not None:
        cell = find_cell(matrix, args.only)
        run_cell(cell, settings=load_settings())
        return
    run_matrix(matrix, skip=args.skip)


if __name__ == "__main__":
    main()
