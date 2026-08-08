"""Versioned system prompts for the four agents.

Each agent owns a registry mapping a version string (the corresponding
`Settings.<agent>_prompt_version` field) to prompt text. `build_<agent>_prompt`
looks its version up and raises `KeyError` if it is missing -- an
unregistered version is a hard error, never a silent fallback to whatever
version happens to be registered. The Researcher, Critic and Supervisor
registries are empty until stages 3, 4 and 5 register their first entry;
every name they will need is already exported here so `config.py`'s
re-export does not change shape as they fill in.
"""

from __future__ import annotations

from tools import PLANNER_TOOLS

_planner_tool_names = ", ".join(tool.name for tool in PLANNER_TOOLS)

_P1 = f"""You are the Planner in a multi-agent research system.

Your job is to turn a user's research request into a structured research
plan, not to answer the request yourself.

Tools available to you: {_planner_tool_names}. Before decomposing the
request, run one or two exploratory searches with these tools first to
understand the domain -- what terminology is used, whether the topic is
covered by the local knowledge base, and what a useful decomposition looks
like. Do not skip this reconnaissance step.

Once you understand the domain, produce a plan with:
- goal: what the research is trying to answer, stated as one clear sentence.
- search_queries: specific, concrete queries the Researcher should run --
  not a restatement of the user's request.
- sources_to_check: which of "knowledge_base", "web", or both the Researcher
  should use, based on what your reconnaissance search found.
- output_format: what the final report should look like (e.g. a comparison
  table, a narrative summary, a ranked list).

Keep the plan concrete and actionable. The Researcher only sees this plan,
not the original conversation."""

PLANNER_PROMPTS: dict[str, str] = {"p1": _P1}
RESEARCHER_PROMPTS: dict[str, str] = {}
CRITIC_PROMPTS: dict[str, str] = {}
SUPERVISOR_PROMPTS: dict[str, str] = {}


def build_planner_prompt(version: str) -> str:
    """Look up the Planner's system prompt by version.

    Parameters
    ----------
    version : str
        A key of `PLANNER_PROMPTS`, normally `Settings.planner_prompt_version`.

    Returns
    -------
    str
        The registered prompt text.

    Raises
    ------
    KeyError
        If `version` is not registered.
    """
    return _lookup("planner", PLANNER_PROMPTS, version)


def build_researcher_prompt(version: str) -> str:
    """Look up the Researcher's system prompt by version.

    See Also
    --------
    build_planner_prompt : Same lookup contract.
    """
    return _lookup("researcher", RESEARCHER_PROMPTS, version)


def build_critic_prompt(version: str) -> str:
    """Look up the Critic's system prompt by version.

    See Also
    --------
    build_planner_prompt : Same lookup contract.
    """
    return _lookup("critic", CRITIC_PROMPTS, version)


def build_supervisor_prompt(version: str) -> str:
    """Look up the Supervisor's system prompt by version.

    See Also
    --------
    build_planner_prompt : Same lookup contract.
    """
    return _lookup("supervisor", SUPERVISOR_PROMPTS, version)


def _lookup(agent_name: str, registry: dict[str, str], version: str) -> str:
    try:
        return registry[version]
    except KeyError:
        raise KeyError(
            f"No {agent_name} prompt registered for version {version!r}. "
            f"Known versions: {sorted(registry)}"
        ) from None
