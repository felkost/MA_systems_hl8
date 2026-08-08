"""LangSmith tracing setup.

`Settings` decides whether tracing is on, not the surrounding environment.
The decision is written to `os.environ`, because that is the only place the
LangSmith SDK reads it from.
"""

from __future__ import annotations

import os
from typing import Any, cast

from langsmith import utils as langsmith_utils
from pydantic import SecretStr

from config import Settings

# The SDK reads `TRACING_V2` before `TRACING`, in both the LANGSMITH and the
# LANGCHAIN namespace. All four names must be written: a leftover
# `LANGCHAIN_TRACING_V2=true` from another project would otherwise keep
# tracing on after this module has turned it off.
TRACING_FLAGS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)

_CREDENTIALS = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_WORKSPACE_ID",
)


def configure_tracing(settings: Settings) -> bool:
    """Export the tracing decision and return it.

    Parameters
    ----------
    settings : Settings
        Configuration for this run. `langsmith_tracing` requests tracing;
        `langsmith_api_key` is what makes it possible.

    Returns
    -------
    bool
        Whether tracing is on. `False` when tracing was requested without an
        API key: refusing the request is easier to diagnose than a background
        thread that fails on every span.

    Notes
    -----
    pydantic-settings reads `.env` without exporting the values, and the SDK
    reads `os.environ` only. Without this copy, a valid key in `.env` produces
    no traces and no error message.

    Real environment variables take priority over `.env` in `Settings`, so
    `LANGSMITH_TRACING=false python main.py` disables tracing for one run.
    """
    api_key = settings.langsmith_api_key if settings.langsmith_tracing else None
    enabled = api_key is not None

    for flag in TRACING_FLAGS:
        os.environ[flag] = "true" if enabled else "false"

    if api_key is not None:
        _export_credentials(settings, api_key)
    else:
        for name in _CREDENTIALS:
            os.environ.pop(name, None)

    # The SDK caches each variable when it first reads it, so an assignment
    # made afterwards has no effect without this call.
    cast(Any, langsmith_utils.get_env_var).cache_clear()
    return enabled


def _export_credentials(settings: Settings, api_key: SecretStr) -> None:
    os.environ["LANGSMITH_API_KEY"] = api_key.get_secret_value()
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    # LangSmith sends this as X-Tenant-Id and rejects an organization-scoped
    # key without it. A personal key does not need it.
    workspace_id = settings.langsmith_workspace_id
    if workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
    else:
        os.environ.pop("LANGSMITH_WORKSPACE_ID", None)
