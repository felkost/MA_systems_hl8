"""Filesystem locations, anchored to the checkout rather than to the caller.

`Settings` stores its paths as relative strings (`data`, `index`,
`.cache/models`) so that no machine-specific path reaches `.env`. Resolving
them here means `python ingest.py` writes the same index from any working
directory, and the model cache stays inside the checkout.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve(path: str | Path) -> Path:
    """Anchor a configured path to the project root.

    Parameters
    ----------
    path : str or Path
        Value of a `Settings` path field. An absolute path is returned
        unchanged, so tests can point at a temporary directory.

    Returns
    -------
    Path
        Absolute path.

    Examples
    --------
    >>> resolve("index") == PROJECT_ROOT / "index"
    True
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def manifest_path(index_dir: str | Path) -> Path:
    """Locate the manifest describing how the index was built.

    See Also
    --------
    corpus_path : The chunk dump BM25 reads.
    """
    return resolve(index_dir) / "manifest.json"


def corpus_path(index_dir: str | Path) -> Path:
    """Locate the chunk dump.

    Notes
    -----
    BM25 scores raw text and cannot read Chroma's vectors, so the chunks are
    also stored as plain JSON.
    """
    return resolve(index_dir) / "chunks.json"
