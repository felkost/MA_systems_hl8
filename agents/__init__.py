"""Sub-agent factories, re-exported under one package name.

`research.py` and `critic.py` join this package at stages 3 and 4.
"""

from __future__ import annotations

from agents.planner import create_planner_agent

__all__ = ["create_planner_agent"]
