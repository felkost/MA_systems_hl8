"""Sub-agent factories, re-exported under one package name.

`critic.py` joins this package at stage 4.
"""

from __future__ import annotations

from agents.planner import create_planner_agent
from agents.research import create_research_agent

__all__ = ["create_planner_agent", "create_research_agent"]
