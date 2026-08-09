"""Build the knowledge index from `data/`.

Run as a separate command: `python ingest.py`. The index is built once and
read many times, so retrieval never triggers embedding of `data/`.

A repeated run is free: chunk ids are derived from the file contents, so a
source file that is already indexed is skipped instead of embedded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neo4j
import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from pydantic import ValidationError

import paths
from config import Settings, load_settings
from graph import (
    GRAPH_EXTRACTION_CONCURRENCY,
    GraphUnavailableError,
    extract_triples_batch,
    extract_triples_freeform,
    get_driver,
    write_triples,
)
from tracing import configure_tracing

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIX = ".pdf"

# Chunks sampled for the schema-vs-free-form relation-name diversity
# measurement. Small on purpose:
# it doubles the LLM calls for the chunks it covers, and the comparison only
# needs enough chunks to show whether free extraction actually diverges.
GRAPH_DIVERSITY_SAMPLE_SIZE = 20


@dataclass(frozen=True)
class IngestStats:
    """What one ingestion run did.

    Attributes
    ----------
    files, pages, chunks : int
        Source files read, page-level documents loaded, chunks in the index.
    added : int
        Chunks embedded and written during this run. Zero means every source
        file was already indexed and unchanged.
    seconds : float
        Wall time of the run.
    """

    files: int
    pages: int
    chunks: int
    added: int
    seconds: float


@traceable(run_type="chain", name="ingest")
def ingest(settings: Settings, embeddings: Embeddings | None = None) -> IngestStats:
    """Turn the documents in `data_dir` into a searchable index.

    Parameters
    ----------
    settings : Settings
        Validated configuration. `data_dir`, `index_dir`, `collection_name`,
        `chunk_size`, `chunk_overlap` and the embedding fields are read here.
    embeddings : Embeddings, optional
        Embedding backend. Defaults to OpenAI. Passing one explicitly allows
        ingestion to run without network access.

    Returns
    -------
    IngestStats
        Counts and wall time of the run.

    Raises
    ------
    FileNotFoundError
        If the data directory is missing or holds no supported document.

    Notes
    -----
    `@traceable` is required here: `Embeddings` is not a `Runnable` and has no
    callbacks, so without it ingestion sends nothing to LangSmith, whatever
    the tracing settings are.
    """
    started = time.perf_counter()
    index_dir = paths.resolve(settings.index_dir)

    documents = _load_documents(paths.resolve(settings.data_dir))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    index_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or _embeddings(settings),
        persist_directory=str(index_dir),
    )
    added = _synchronize(store, chunks)

    files = len({document.metadata["source"] for document in documents})
    _write_corpus(chunks, paths.corpus_path(index_dir))
    _write_manifest(settings, files, len(chunks), paths.manifest_path(index_dir))

    return IngestStats(
        files=files,
        pages=len(documents),
        chunks=len(chunks),
        added=added,
        seconds=round(time.perf_counter() - started, 2),
    )


@dataclass(frozen=True)
class GraphIngestStats:
    """What one `ingest.py --graph` run did.

    Attributes
    ----------
    chunks : int
        Chunks read from the corpus dump and offered to extraction.
    triples_written : int
        Relations merged into the graph -- one `write_triples` call per
        chunk's extracted triples, summed.
    schema_relation_types, freeform_relation_types : int
        Distinct predicate strings seen on `diversity_sample_size` chunks,
        under the closed schema and under no constraint.
    diversity_sample_size : int
        How many chunks the diversity comparison covered -- doubles their
        LLM calls, so it stays a sample, not the whole corpus.
    seconds : float
        Wall time of the run.
    """

    chunks: int
    triples_written: int
    schema_relation_types: int
    freeform_relation_types: int
    diversity_sample_size: int
    seconds: float


@traceable(run_type="chain", name="ingest_graph")
def ingest_graph(
    settings: Settings,
    model: BaseChatModel | None = None,
    driver: neo4j.Driver | None = None,
) -> GraphIngestStats:
    """Extract entities and relations from the corpus into the graph.

    Parameters
    ----------
    settings : Settings
        `neo4j_database` and the model fields are read here.
    model : BaseChatModel, optional
        Extraction model. Defaults to `ChatOpenAI` built from `settings`.
        Passing one explicitly allows this to run without an API call.
    driver : neo4j.Driver, optional
        Defaults to `graph.get_driver()`. Passing one explicitly allows this
        to run without a Neo4j container.

    Returns
    -------
    GraphIngestStats
        Counts and wall time of the run.

    Raises
    ------
    FileNotFoundError
        If the corpus dump `ingest()` writes is missing -- run the plain
        ingest first.
    GraphUnavailableError
        If Neo4j is not configured and no `driver` was passed.

    Notes
    -----
    Reads the same `chunks.json` the lexical retriever reads, rather than
    re-parsing `data_dir`: `ingest()` already wrote it in this run or an
    earlier one, and re-parsing would extract from text that may not match
    what is actually indexed.
    """
    started = time.perf_counter()
    chunks = _load_corpus_chunks(settings)

    extraction_model = model or ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        temperature=0,
    )
    graph_driver = driver or get_driver()
    sample_size = min(len(chunks), GRAPH_DIVERSITY_SAMPLE_SIZE)

    triples_written = 0
    schema_predicates: set[str] = set()
    freeform_predicates: set[str] = set()
    window_size = GRAPH_EXTRACTION_CONCURRENCY
    for start in range(0, len(chunks), window_size):
        window = chunks[start : start + window_size]
        texts = [str(chunk["text"]) for chunk in window]
        extracted = extract_triples_batch(
            texts, extraction_model, max_concurrency=window_size
        )

        for offset, (chunk, triples) in enumerate(zip(window, extracted)):
            metadata = chunk["metadata"]
            source = f"{metadata['source']}|{metadata['page']}"

            schema_predicates.update(triple.predicate for triple in triples)
            # Writes stay sequential: they are local and fast next to a model
            # round trip, and one writer keeps the MERGE order predictable.
            triples_written += write_triples(
                graph_driver, triples, source, settings.neo4j_database
            )

            if start + offset < sample_size:
                free_triples = extract_triples_freeform(texts[offset], extraction_model)
                freeform_predicates.update(
                    free_triple.predicate.strip().lower()
                    for free_triple in free_triples
                )

    return GraphIngestStats(
        chunks=len(chunks),
        triples_written=triples_written,
        schema_relation_types=len(schema_predicates),
        freeform_relation_types=len(freeform_predicates),
        diversity_sample_size=sample_size,
        seconds=round(time.perf_counter() - started, 2),
    )


def _load_corpus_chunks(settings: Settings) -> list[dict[str, Any]]:
    path = paths.corpus_path(settings.index_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"no chunk dump at {path} -- run `python ingest.py` first"
        )
    payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _load_documents(data_dir: Path) -> list[Document]:
    """Read every supported file as one document per page.

    Metadata is rebuilt instead of reused: pypdf returns a producer, a
    creation date and other values that differ between machines, and Chroma
    rejects the `None` values among them. `source` is the file name, not the
    full path, so a chunk id does not depend on the location of the checkout.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {data_dir}. Put the source documents "
            "there, or point DATA_DIR at the right place."
        )

    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == PDF_SUFFIX:
            documents.extend(_read_pdf(path))
        elif suffix in TEXT_SUFFIXES:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name, "page": 0},
                )
            )

    if not documents:
        raise FileNotFoundError(f"no .pdf, .txt or .md files in {data_dir}")
    return documents


def _read_pdf(path: Path) -> list[Document]:
    reader = pypdf.PdfReader(str(path))
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": number},
        )
        for number, page in enumerate(reader.pages)
    ]


def _chunk_id(chunk: Document) -> str:
    """Derive a stable id from the chunk's provenance and its text.

    Without explicit ids Chroma assigns a new UUID on every run and duplicates
    the whole corpus. Retrieval then returns the same passage several times,
    and nothing reports an error.

    The chunk text is part of the key, not only its position. An edit that
    does not move the offsets after it -- a corrected word, a replaced
    sentence -- leaves `source|page|start_index` unchanged, so an id built
    from the position alone would mark the new text as already indexed and the
    old text would stay in the index.
    """
    meta = chunk.metadata
    key = (
        f"{meta['source']}|{meta['page']}|{meta['start_index']}"
        f"|{chunk.page_content}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _synchronize(store: Chroma, chunks: list[Document]) -> int:
    """Bring the collection in line with `chunks`, file by file.

    Returns the number of chunks embedded. Chroma embeds everything passed to
    `add_documents`, with or without ids, so a repeated run is only free if
    whole unchanged files are skipped. A file whose chunk set changed is
    replaced completely: an edit moves every offset after it, and adding the
    new chunks without deleting the old ones would leave the previous text in
    the index, where it can still be retrieved.

    A source file removed from the input entirely is pruned by
    `_prune_removed_sources`: this loop only ever visits sources present in
    `chunks`, so a source that disappeared from `data_dir` would otherwise
    never be revisited, let alone deleted.
    """
    added = 0
    current_sources: set[str] = set()
    for source, source_chunks in _by_source(chunks).items():
        current_sources.add(source)
        wanted = [_chunk_id(chunk) for chunk in source_chunks]
        stored = store.get(where={"source": source}, include=[])["ids"]
        if set(stored) == set(wanted):
            continue
        if stored:
            store.delete(ids=stored)
        store.add_documents(source_chunks, ids=wanted)
        added += len(wanted)
    _prune_removed_sources(store, current_sources)
    return added


def _prune_removed_sources(store: Chroma, current_sources: set[str]) -> None:
    """Delete every chunk whose source file is no longer in `data_dir`.

    A deleted PDF must not go on answering `knowledge_search` forever just
    because nothing else in `_synchronize` looks for a source that vanished
    from the current `chunks` rather than merely changing.
    """
    stored_sources = {
        metadata["source"] for metadata in store.get(include=["metadatas"])["metadatas"]
    }
    for source in stored_sources - current_sources:
        stale_ids = store.get(where={"source": source}, include=[])["ids"]
        if stale_ids:
            store.delete(ids=stale_ids)


def _by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata["source"]].append(chunk)
    return grouped


def _write_corpus(chunks: list[Document], path: Path) -> None:
    payload = [
        {
            "id": _chunk_id(chunk),
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_manifest(
    settings: Settings,
    files: int,
    chunks: int,
    path: Path,
) -> None:
    """Record what the index was built with.

    `retriever.py` refuses an index whose manifest does not match the current
    settings. A different embedding model or vector width does not raise an
    error on its own; it returns wrong results.
    """
    manifest = {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "files": files,
        "chunks": chunks,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _embeddings(settings: Settings) -> Embeddings:
    arguments: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.api_key,
    }
    if settings.embedding_dimensions is not None:
        arguments["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**arguments)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the knowledge index from data/."
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also extract entities and relations into the Neo4j knowledge graph.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    try:
        settings = load_settings()
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY and values in .env.")
        raise SystemExit(1)

    configure_tracing(settings)
    print(f"Reading {paths.resolve(settings.data_dir)}")

    try:
        stats = ingest(settings)
    except FileNotFoundError as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1)

    print(
        f"Indexed {stats.chunks} chunks from {stats.pages} pages "
        f"of {stats.files} files in {stats.seconds}s "
        f"({stats.added} embedded this run)."
    )
    print(f"Index: {paths.resolve(settings.index_dir)}")

    if args.graph:
        try:
            graph_stats = ingest_graph(settings)
        except GraphUnavailableError as error:
            print(f"Graph ingestion skipped: {error}")
            raise SystemExit(1)
        print(
            f"Graph: {graph_stats.triples_written} relations from "
            f"{graph_stats.chunks} chunks in {graph_stats.seconds}s. "
            f"Relation-name diversity on a sample of "
            f"{graph_stats.diversity_sample_size} chunks -- "
            f"schema: {graph_stats.schema_relation_types}, "
            f"free-form: {graph_stats.freeform_relation_types}."
        )


if __name__ == "__main__":
    main()
