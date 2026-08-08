"""Knowledge graph over entities and relations extracted from the corpus.

Built by `ingest.py --graph`, read by the `graph_search` tool (offered to the
Researcher only). Both the relation types and the entity types are a closed
enum rather than free text: an unconstrained extraction fragments the same
real-world entity into several spellings and invents a new relation name for
every synonym a source happens to use. `extract_triples_freeform` exists only
to measure that difference as `ingest.py`'s own stage artifact -- nothing in
the graph itself is built from its output.

Entities are normalized to a lowercase, article-stripped record before they
reach Neo4j, and every write is `MERGE`, not `CREATE`, so both extraction and
writing stay idempotent: running `ingest.py --graph` again does not duplicate
a relation the graph already has.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal, get_args

import neo4j
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from config import Settings, load_settings

EntityType = Literal["MODEL", "TECHNOLOGY", "FRAMEWORK", "CONCEPT", "COMPANY", "PERSON"]
Predicate = Literal[
    "IS_A",
    "PART_OF",
    "USES",
    "IMPLEMENTS",
    "COMPARES_TO",
    "IMPROVES_ON",
    "TRAINED_ON",
]

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


class Triple(BaseModel):
    """One schema-constrained relation, as the model extracted it.

    Entity names are kept as written here; `write_triples` normalizes them on
    the way into the graph, so a test can assert on the extraction result
    without also asserting on normalization.
    """

    subject: str
    subject_type: EntityType
    predicate: Predicate
    object: str
    object_type: EntityType


class ExtractedTriples(BaseModel):
    """Structured-output envelope.

    `with_structured_output` builds a JSON schema from its argument, and a
    bare `list[Triple]` has no schema of its own to build one from.
    """

    triples: list[Triple] = Field(default_factory=list)


class FreeTriple(BaseModel):
    """One relation extracted with no constraint on the predicate string.

    Exists only for the diversity measurement `ingest.py` logs: how many
    distinct relation names an unconstrained extraction invents, against how
    many the closed `Predicate` enum allows.
    """

    subject: str
    predicate: str
    object: str


class ExtractedFreeTriples(BaseModel):
    triples: list[FreeTriple] = Field(default_factory=list)


class GraphUnavailableError(RuntimeError):
    """The graph cannot be reached, or Neo4j is not configured."""


_EXTRACTION_PROMPT = """Extract factual relations between named entities from \
the passage below.

Use only these relation types: {predicates}.
Use only these entity types: {entity_types}.

Extract only relations the passage actually states. Return no triples rather \
than a guessed one.

Passage:
{text}"""

_FREEFORM_EXTRACTION_PROMPT = """Extract factual relations between named \
entities from the passage below, as subject-predicate-object triples. \
Describe each predicate in your own words -- there is no fixed list of \
relation types to choose from.

Extract only relations the passage actually states. Return no triples rather \
than a guessed one.

Passage:
{text}"""


def normalize_entity(name: str) -> str:
    """Fold an entity name to the record Neo4j stores.

    Two spellings of the same real-world entity -- "the Transformer" from one
    chunk, "Transformer" from another -- must `MERGE` onto the same node, or
    the graph fragments into near-duplicate entities the way an unconstrained
    extraction fragments relation names.
    """
    stripped = _ARTICLES.sub("", name.strip())
    return " ".join(stripped.lower().split())


def extract_triples(text: str, model: BaseChatModel) -> list[Triple]:
    """Extract schema-constrained relations from one chunk of text.

    Parameters
    ----------
    text : str
        Chunk text to extract relations from.
    model : BaseChatModel
        Chat model bound to `ExtractedTriples` through `with_structured_output`.

    Returns
    -------
    list of Triple
        Empty when the passage states no relation the schema can represent.
    """
    prompt = _EXTRACTION_PROMPT.format(
        predicates=", ".join(get_args(Predicate)),
        entity_types=", ".join(get_args(EntityType)),
        text=text,
    )
    result = model.with_structured_output(ExtractedTriples).invoke(prompt)
    # with_structured_output can return a bare dict when the caller asks for
    # the raw response alongside the parsed one; this call never does, so the
    # assert is a real narrowing, not a suppression.
    assert isinstance(result, ExtractedTriples)
    return result.triples


def extract_triples_freeform(text: str, model: BaseChatModel) -> list[FreeTriple]:
    """Extract relations with an unconstrained predicate string.

    See Also
    --------
    extract_triples : The schema-constrained extraction the graph is built
        from. This function feeds only the diversity measurement in
        `ingest.py`.
    """
    prompt = _FREEFORM_EXTRACTION_PROMPT.format(text=text)
    result = model.with_structured_output(ExtractedFreeTriples).invoke(prompt)
    assert isinstance(result, ExtractedFreeTriples)
    return result.triples


def build_driver(settings: Settings) -> neo4j.Driver:
    """Build a Neo4j driver from `settings`.

    Raises
    ------
    GraphUnavailableError
        If `neo4j_password` is unset. Every agent still starts without it --
        only `graph_search` and `ingest.py --graph` need the graph -- the
        same shape as a missing index only disabling `knowledge_search`.

    Notes
    -----
    `GraphDatabase.driver` parses the URI but does not connect; a bad host or
    a stopped container only surfaces once a query actually runs.
    """
    if settings.neo4j_password is None:
        raise GraphUnavailableError(
            "Neo4j is not configured -- set NEO4J_PASSWORD in .env"
        )
    return neo4j.GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )


@lru_cache(maxsize=1)
def get_driver() -> neo4j.Driver:
    """Build the driver once and hand the same one to every caller.

    See Also
    --------
    retriever.get_retriever : The same cached-singleton shape, for the same
        reason -- building the underlying client on every tool call is
        wasted work the cache avoids for the life of the process.
    """
    return build_driver(load_settings())


_MERGE_QUERY = """
MERGE (s:Entity {{name: $subject}})
  ON CREATE SET s.type = $subject_type
MERGE (o:Entity {{name: $object}})
  ON CREATE SET o.type = $object_type
MERGE (s)-[r:{predicate}]->(o)
  ON CREATE SET r.sources = [$source]
  ON MATCH SET r.sources = CASE WHEN $source IN r.sources
    THEN r.sources ELSE r.sources + $source END
"""


def write_triples(
    driver: neo4j.Driver,
    triples: list[Triple],
    source: str,
    database: str,
) -> int:
    """Write `triples` into the graph, idempotently.

    Parameters
    ----------
    driver : neo4j.Driver
        Open driver, from `get_driver` or a fake in a test.
    triples : list of Triple
        Already schema-validated relations.
    source : str
        Provenance recorded on each relation, e.g. `"paper.pdf|3"`. Appended
        to the relation's `sources` list rather than replacing it, so the
        same fact seen in two chunks keeps both.
    database : str
        Neo4j database name, `settings.neo4j_database`.

    Returns
    -------
    int
        Number of triples written. Every call is one `MERGE`, so re-running
        this over the same triples changes nothing in the graph.

    Notes
    -----
    `predicate` is interpolated into the query text rather than bound as a
    parameter, because Cypher does not allow a parameter as a relationship
    type. This is not a Cypher-injection risk: `Triple.predicate` is a
    `pydantic` `Literal`, already validated against the closed `Predicate`
    enum before an instance can exist.
    """
    written = 0
    with driver.session(database=database) as session:
        for triple in triples:
            session.run(
                _MERGE_QUERY.format(predicate=triple.predicate),
                subject=normalize_entity(triple.subject),
                subject_type=triple.subject_type,
                object=normalize_entity(triple.object),
                object_type=triple.object_type,
                source=source,
            )
            written += 1
    return written


_SEARCH_QUERY = """
MATCH (e:Entity)-[r]-(other:Entity)
WHERE toLower(e.name) CONTAINS $entity
RETURN e.name AS entity, type(r) AS relation, other.name AS other,
       other.type AS other_type
LIMIT $limit
"""


def query_entity(
    driver: neo4j.Driver,
    entity: str,
    database: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find how `entity` relates to others in the graph.

    Parameters
    ----------
    driver : neo4j.Driver
        Open driver.
    entity : str
        Entity name as the caller wrote it; normalized the same way ingestion
        normalized it, so casing and articles do not stop a match.
    database : str
        Neo4j database name.
    limit : int, default=20
        Relations returned. Enough for the model to choose the next hop
        itself -- this function does not traverse a second hop on its own.

    Returns
    -------
    list of dict
        One row per relation: `entity`, `relation`, `other`, `other_type`.
        Empty when the entity is not in the graph.
    """
    with driver.session(database=database) as session:
        result = session.run(
            _SEARCH_QUERY,
            entity=normalize_entity(entity),
            limit=limit,
        )
        return [record.data() for record in result]
