"""The tools an agent can call.

Each tool truncates large output and returns a failure as a string starting
with ``ERROR:`` instead of raising. The model reads that string and can react
to it, so a failed tool does not end the run.

Tool docstrings are read by the model: LangChain sends the whole docstring as
the tool description. They therefore stay short and say when to use the tool,
instead of following the numpydoc layout used in the rest of the code.

The four named lists at the bottom are the single source of truth for what
each agent can call; the corresponding agent factory builds its tool list
from here, and every name must also appear in that agent's system prompt, or
the model never learns that the tool exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import httpx
import trafilatura
from ddgs import DDGS
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from config import Settings, load_settings
from graph import GraphUnavailableError, get_driver, query_entity
from retriever import IndexMismatchError, get_retriever

SearchResult = TypedDict(
    "SearchResult",
    {
        "title": str,
        "url": str,
        "snippet": str,
    },
)


@tool
def web_search(query: str) -> list[SearchResult] | str:
    """Search the web and return compact candidate sources.

    Use this tool to discover pages relevant to a research
    question. Search snippets are not full source texts.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return "ERROR: Search query cannot be empty."

    try:
        settings = load_settings()
        raw_results = DDGS().text(
            normalized_query,
            max_results=settings.max_search_results,
        )
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            title = str(item.get("title") or "Untitled").strip()
            url = str(item.get("href") or "").strip()
            snippet = str(item.get("body") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet[: settings.max_search_snippet_length],
                }
            )
        return results
    except Exception:
        # Exception text can contain DNS names and local paths, and it would
        # reach the model unchanged. Report the failure without the details.
        return "ERROR: Web search is temporarily unavailable."


@tool
def read_url(url: str) -> str:
    """Read the main text content of an HTTP or HTTPS page.

    Use this tool after web_search when a source needs to be
    examined in detail.
    """
    normalized_url = url.strip()
    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        return "ERROR: URL must be a valid HTTP or HTTPS address."
    try:
        settings = load_settings()
        response = httpx.get(
            normalized_url,
            timeout=settings.http_timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        extracted_text = trafilatura.extract(response.text)
        if not extracted_text:
            return "ERROR: No readable text was found on the page."
        text = extracted_text.strip()
        if len(text) <= settings.max_url_content_length:
            return text
        truncated_text = text[: settings.max_url_content_length]
        return (
            f"{truncated_text}\n\n"
            f"[Content truncated to "
            f"{settings.max_url_content_length} characters.]"
        )
    except httpx.TimeoutException:
        return "ERROR: The page request timed out."
    except httpx.HTTPError:
        return "ERROR: The page is unavailable."
    except Exception:
        return "ERROR: The page could not be read."


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base built from the documents in data/.

    Prefer this over web_search for anything the ingested documents might
    cover: it is faster, always available, and each result names a stable
    source and page.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return "ERROR: Search query cannot be empty."

    try:
        documents = get_retriever().invoke(normalized_query)
    except (FileNotFoundError, IndexMismatchError) as error:
        return f"ERROR: {error}"
    except Exception:
        # The cause can be a raw model or I/O error, potentially naming a
        # local path. Report the failure without the details.
        return "ERROR: Knowledge base search is unavailable."

    if not documents:
        return "No matching passages in the knowledge base."

    return _format_passages(documents, load_settings())


def _format_passages(documents: list[Document], settings: Settings) -> str:
    count = len(documents)
    header = f"[{count} document{'s' if count != 1 else ''} found]"
    passages = "\n".join(
        f"- [{document.metadata.get('source', 'unknown')}, "
        f"page {document.metadata.get('page', 0) + 1}] "
        f"{document.page_content.strip()}"
        for document in documents
    )

    top_score = documents[0].metadata.get("rerank_score")
    warning = ""
    if top_score is not None and top_score < settings.rerank_confidence_floor:
        warning = (
            "\n[low confidence - the knowledge base may not cover this well; "
            "consider rephrasing the query or using web_search]"
        )

    text = f"{header}\n{passages}{warning}"
    if len(text) <= settings.max_knowledge_search_length:
        return text
    truncated = text[: settings.max_knowledge_search_length]
    return (
        f"{truncated}\n\n"
        f"[Content truncated to "
        f"{settings.max_knowledge_search_length} characters.]"
    )


@tool
def graph_search(entity: str) -> str:
    """Find how an entity relates to others in the knowledge graph.

    Use after knowledge_search when the question spans several documents:
    "which X was used by the Y that does Z". Call again with a name found in
    the results to follow the chain one more hop.
    """
    normalized_entity = entity.strip()
    if not normalized_entity:
        return "ERROR: Entity name cannot be empty."

    try:
        settings = load_settings()
        rows = query_entity(get_driver(), normalized_entity, settings.neo4j_database)
    except GraphUnavailableError as error:
        return f"ERROR: {error}"
    except Exception:
        # The cause can be a driver or server error naming a host or a local
        # path. Report the failure without the details.
        return "ERROR: Knowledge graph is unavailable."

    if not rows:
        return f"No relations found for {normalized_entity!r} in the knowledge graph."

    return _format_graph_rows(rows, settings)


def _format_graph_rows(rows: list[dict[str, Any]], settings: Settings) -> str:
    count = len(rows)
    header = f"[{count} relation{'s' if count != 1 else ''} found]"
    lines = "\n".join(
        f"- {row['entity']} {row['relation']} {row['other']} ({row['other_type']})"
        for row in rows
    )

    text = f"{header}\n{lines}"
    if len(text) <= settings.max_graph_search_length:
        return text
    truncated = text[: settings.max_graph_search_length]
    return (
        f"{truncated}\n\n"
        f"[Content truncated to "
        f"{settings.max_graph_search_length} characters.]"
    )


@tool
def save_report(filename: str, content: str) -> str:
    """Save a completed Markdown research report.

    The report is written as a .md file inside the configured
    output directory. Only call this once the Critic has approved
    the findings -- a human must still approve the write itself.
    """
    if not content.strip():
        return "ERROR: Report content cannot be empty."

    normalized_name = filename.strip().replace("\\", "/")
    base_name = normalized_name.rsplit("/", maxsplit=1)[-1]
    stem = Path(base_name).stem
    # re.UNICODE keeps Cyrillic in the name, so a Ukrainian filename does not
    # collapse to an empty stem.
    safe_stem = re.sub(r"[^\w.-]", "", stem, flags=re.UNICODE).strip(".")
    if not safe_stem:
        return "ERROR: Report filename is invalid."

    try:
        settings = load_settings()
        output_directory = Path(settings.output_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        report_path = (output_directory / f"{safe_stem}.md").resolve()
        if report_path.parent != output_directory:
            return "ERROR: Report path is outside the output directory."

        report_path.write_text(content, encoding="utf-8")
        return f"Report saved to: {report_path}"
    except Exception:
        return "ERROR: Report could not be saved."


# Each agent gets exactly the tools its architecture row prescribes. The
# Planner needs breadth, not a deep-read or graph-hop tool; graph_search is
# routed to the Researcher only, never Planner or Critic, so a fourth tool on
# the Critic cannot widen the surface on which it rubber-stamps a verdict.
# SUPERVISOR_TOOLS covers only the tool this module defines for the
# Supervisor -- `supervisor.py` extends it with the `plan`/`research`/
# `critique` agent-as-tool wrappers once those exist.
PLANNER_TOOLS: list[BaseTool] = [web_search, knowledge_search]
RESEARCHER_TOOLS: list[BaseTool] = [
    web_search,
    read_url,
    knowledge_search,
    graph_search,
]
CRITIC_TOOLS: list[BaseTool] = [web_search, read_url, knowledge_search]
SUPERVISOR_TOOLS: list[BaseTool] = [save_report]
