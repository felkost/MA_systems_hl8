"""Two-stage retrieval over the index that `ingest.py` builds.

The first stage merges a semantic arm (Chroma) with a lexical arm (BM25) and
aims at recall: documents it misses cannot be recovered later, because the
reranker only reorders what it receives. The second stage is a cross encoder
that reads query and passage together and aims at precision, cutting the
merged candidates down to `rerank_top_n`.

Both stages are LangChain retrievers rather than plain functions.
`BaseRetriever.invoke` emits a LangSmith span with the query and the
documents, so a trace shows the candidate list before and after reranking. A
local `rerank(documents, query)` would give the same ranking but no record of
it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import (
    BaseCrossEncoder,
    HuggingFaceCrossEncoder,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

import paths
from config import Settings, load_settings

# Manifest fields that make the stored vectors unusable when they differ.
# `chunk_size` and `chunk_overlap` are recorded next to them but are not
# checked: chunks written at one size still work at another, only with
# different boundaries.
INCOMPATIBLE_FIELDS = (
    "embedding_model",
    "embedding_dimensions",
    "collection_name",
)

BUILD_HINT = "run `python ingest.py` first"


class IndexMismatchError(RuntimeError):
    """The index on disk cannot answer questions under these settings."""


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """`CrossEncoderReranker` that keeps the score on the documents it keeps.

    The stock implementation scores every candidate, sorts and truncates to
    `top_n`, then returns plain `Document` objects -- the score itself is
    discarded, confirmed by reading its source. `tools.knowledge_search`
    needs that number to tell the model when even the best match is weak.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        scores = self.model.score(
            [(query, document.page_content) for document in documents]
        )
        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        kept = []
        for document, score in ranked[: self.top_n]:
            document.metadata = {**document.metadata, "rerank_score": score}
            kept.append(document)
        return kept


def configure_model_cache(settings: Settings) -> Path:
    """Point the Hugging Face and torch caches inside the checkout.

    Parameters
    ----------
    settings : Settings
        `model_cache_dir` is read here; a relative value is anchored to the
        project root.

    Returns
    -------
    Path
        The cache root, already created.

    Notes
    -----
    The variables are overwritten, not set as defaults: an `HF_HOME` left by
    another project would otherwise decide where a large model is downloaded.

    `HF_HOME` alone is not enough, which is why `_cross_encoder` also passes
    `cache_folder`. `huggingface_hub` computes its cache location once, when it
    is imported, and importing this module already imports it through
    `langchain_community.cross_encoders`. A value written here therefore
    arrives too late for that library. `TORCH_HOME` is read on use and does
    apply, and both variables still reach subprocesses.
    """
    cache = paths.resolve(settings.model_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    return cache


def build_retriever(
    settings: Settings,
    embeddings: Embeddings | None = None,
    cross_encoder: BaseCrossEncoder | None = None,
) -> BaseRetriever:
    """Assemble the ensemble and the reranker over an existing index.

    Parameters
    ----------
    settings : Settings
        `retrieval_top_k`, `rerank_top_n` and `ensemble_bm25_weight` shape the
        funnel; `index_dir` and `collection_name` locate the index.
    embeddings : Embeddings, optional
        Query embedder. Defaults to OpenAI. Passing one explicitly allows the
        retriever to be built without network access.
    cross_encoder : BaseCrossEncoder, optional
        Reranking model. Defaults to `reranker_model` on `reranker_device`.
        Passing one explicitly avoids downloading about 1.1 GB.

    Returns
    -------
    BaseRetriever
        A `ContextualCompressionRetriever`, so a trace shows the candidates
        before and after reranking.

    Raises
    ------
    FileNotFoundError
        If the index or the chunk file is missing.
    IndexMismatchError
        If the index was built with settings that make its vectors unusable,
        or if it holds no chunks at all.

    See Also
    --------
    get_retriever : The cached form that takes no arguments.
    """
    _verify_manifest(settings)
    chunks = _load_chunks(settings)

    semantic = _store(settings, embeddings).as_retriever(
        search_kwargs={"k": settings.retrieval_top_k},
    )
    lexical = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
    lexical.k = settings.retrieval_top_k

    ensemble = EnsembleRetriever(
        retrievers=[lexical, semantic],
        weights=[
            settings.ensemble_bm25_weight,
            1 - settings.ensemble_bm25_weight,
        ],
    )
    reranker = ScoredCrossEncoderReranker(
        model=cross_encoder or _cross_encoder(settings),
        top_n=settings.rerank_top_n,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """Build the retriever once and hand the same one to every caller.

    Notes
    -----
    Loading the cross encoder takes several seconds and about 1.1 GB of
    memory. Building it once per tool call would make every search several
    seconds slower. The cache lasts for the life of the process.
    """
    return build_retriever(load_settings())


def _verify_manifest(settings: Settings) -> dict[str, Any]:
    """Refuse an index these settings cannot read.

    A mismatched index does not raise an error on its own. Vectors from
    another embedding model belong to a different space but still return
    nearest neighbours, and a collection name that does not exist reads as an
    empty collection. Both give wrong answers, so the manifest is checked.
    """
    path = paths.manifest_path(settings.index_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no index manifest at {path} -- {BUILD_HINT}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    differing = [
        f"{field} (index {manifest.get(field)!r}, "
        f"settings {getattr(settings, field)!r})"
        for field in INCOMPATIBLE_FIELDS
        if manifest.get(field) != getattr(settings, field)
    ]
    if differing:
        raise IndexMismatchError(
            "index built with different settings: "
            + ", ".join(differing)
            + f" -- restore the values or rebuild it: {BUILD_HINT}"
        )
    return manifest


def _load_chunks(settings: Settings) -> list[Document]:
    """Read the chunk file that the lexical arm scores.

    BM25 works on text and cannot read Chroma's vectors, so `ingest.py` also
    writes the chunks as JSON. The explicit encoding is required:
    `read_text()` uses the locale of the process, which fails on the first
    non-Latin character of real PDF text.
    """
    path = paths.corpus_path(settings.index_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no chunk dump at {path} -- {BUILD_HINT}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        # BM25Retriever.from_documents([]) raises inside rank_bm25, with a
        # message that does not name the real cause.
        raise IndexMismatchError(f"the chunk dump at {path} is empty -- {BUILD_HINT}")
    return [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in payload
    ]


def _store(settings: Settings, embeddings: Embeddings | None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or _embeddings(settings),
        persist_directory=str(paths.resolve(settings.index_dir)),
    )


def _embeddings(settings: Settings) -> Embeddings:
    """Build the query embedder that matches the one `ingest.py` used.

    Both modules must agree on the model and on the vector width. The manifest
    check enforces that at run time by comparing the recorded values with the
    current settings, which catches an edit to `.env` as well as one to the
    code.
    """
    arguments: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.api_key,
    }
    if settings.embedding_dimensions is not None:
        arguments["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**arguments)


def _cross_encoder(settings: Settings) -> BaseCrossEncoder:
    """Load the reranking model on the configured device and cache directory.

    `model_kwargs` is not Hugging Face's argument of the same name:
    `HuggingFaceCrossEncoder` passes the whole dict to
    `sentence_transformers.CrossEncoder` as keyword arguments, which is how
    `cache_folder` reaches it. That parameter, not `HF_HOME`, decides where the
    model is downloaded -- see `configure_model_cache`.
    """
    cache = configure_model_cache(settings)
    return HuggingFaceCrossEncoder(
        model_name=settings.reranker_model,
        model_kwargs={
            "device": _resolve_device(settings),
            "cache_folder": str(cache / "huggingface"),
        },
    )


def _resolve_device(settings: Settings) -> str:
    """Resolve `auto` to the device this machine actually has.

    `torch` is imported inside the function: a fixed `reranker_device` does not
    need it, and the import itself takes several seconds.
    """
    if settings.reranker_device != "auto":
        return settings.reranker_device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _tokenize(text: str) -> list[str]:
    """Fold case and drop punctuation before BM25 sees the text.

    BM25 matches tokens literally. Without this step "seq2seq", "Seq2seq" and
    "seq2seq," are three different terms, and the lexical arm loses the rare
    exact matches it exists to find.
    """
    return re.findall(r"\w+", text.lower())
