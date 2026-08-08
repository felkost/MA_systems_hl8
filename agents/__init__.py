"""Sub-agent factories, re-exported under one package name."""

from __future__ import annotations

from agents.critic import create_critic_agent
from agents.planner import create_planner_agent
from agents.research import create_research_agent

__all__ = ["create_critic_agent", "create_planner_agent", "create_research_agent"]
