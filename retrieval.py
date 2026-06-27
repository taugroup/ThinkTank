from collections import defaultdict

from logger import logger
from typing import List, Optional

from config import DB_PATH, HNSW_SPACE


def _lazy_retrieval_deps():
    """Import Chroma/embeddings only when a real retrieval is requested.

    Keeps the module importable (and mock mode runnable) without chromadb/ollama.
    """
    from db_connection import ChromaDBConnection
    from ingestion import get_embeddings

    return ChromaDBConnection, get_embeddings

def retrieve_documents(queries: List[str], collection_name: str) -> List[str]:
    """
    Performs vector search on collections with dense retrieval.

    Args:
        queries (List[str]): List of translated user queries.
        collection_name (str): Name of the collection in ChromaDB.

    Returns:
        dict: {
            'query_list': List[str],           # The input queries
            'context': List[str],              # List of document texts
            'retrieved_docs': List[dict]       # List of dicts with 'text' and 'source'
        }
    """
    logger.info("Performing retrieval...")
    ChromaDBConnection, get_embeddings = _lazy_retrieval_deps()
    db = ChromaDBConnection(DB_PATH)
    collection = db.get_collection(name=collection_name, metadata={"hnsw:space": HNSW_SPACE})
    embeddings = get_embeddings()

    logger.info("Fetching all documents from ChromaDB for sparse retrieval...")

    dense_scores = defaultdict(float)
    for query in queries:
        query_vector = embeddings.embed_query(query)
        retrieved = collection.query(query_embeddings=[query_vector], n_results=10, include=["distances"])
        for i, doc_id in enumerate(retrieved["ids"][0]):
            # Convert distance to similarity score: similarity = 1 - distance
            raw_dist = 1 - retrieved["distances"][0][i]
            score = 1 - raw_dist if HNSW_SPACE == "cosine" else -raw_dist
            dense_scores[doc_id] += score

    # top-k by combined dense score
    top_ids = [doc for doc, _ in sorted(dense_scores.items(),
                                        key=lambda x: x[1],
                                        reverse=True)][:10]

    docs = collection.get(ids=top_ids, include=["documents", "metadatas"])
    dense_results = [ {'text': doc_text, 'source': metadata['source']}  for _, doc_text, metadata in zip(top_ids, docs["documents"], docs["metadatas"])]
    logger.info(f"Retrieved {len(dense_results)} documents from collection '{collection_name}'.")

    #Display unique sources in Streamlit app
    unique_sources = set(result['source'] for result in dense_results)
    # st.markdown(f"### References from collection: {collection_name}:")
    # st.session_state.markdown_log.append("### References:")
    # for source in unique_sources:
    #     st.markdown(f"- {source}")
    #     st.session_state.markdown_log.append(f"- {source}")

    # websocket.send_json({'name': 'Retrieved Documents', 'content': f"Retrieved {len(dense_results)} documents from collection '{collection_name}'."})
    # for source in unique_sources:
    #     websocket.send_json({'name': f'- {source}', 'content': ''})


    # Keep full document objects for tool call output
    tool_output = {
        'query_list': queries,
        'context': [result['text'] for result in dense_results],
        'retrieved_docs': dense_results  # Include full document objects with text and source
    }
    return tool_output

# ---------------------------------------------------------------------------
# Policy evidence retrieval (NEW public seam — OWNER: Person 2)
# ---------------------------------------------------------------------------

def _mock_evidence(queries: List[str], geography: Optional[str]):
    """Domain-neutral placeholder evidence for mock/dev mode.

    Generates clearly-labeled PLACEHOLDER EvidenceItems templated to the query and
    geography so the rest of the team is unblocked for ANY policy domain before the
    real Chroma collection exists. These are not real sources."""
    from models import EvidenceItem

    topic = (queries[0] if queries else "the policy question").strip()
    geo = geography or "the target geography"
    return [
        EvidenceItem(
            source_id="PLACEHOLDER-GOV",
            title=f"Government Review Relevant to: {topic}",
            organization="(placeholder) Government Source",
            source_type="government_report",
            publication_date="2023",
            geography=geo,
            page=1,
            text=f"Placeholder government evidence relevant to {topic} in {geo}.",
            relevance_score=0.9,
            credibility_score=0.85,
        ),
        EvidenceItem(
            source_id="PLACEHOLDER-ACADEMIC",
            title=f"Comparative Study Relevant to: {topic}",
            organization="(placeholder) Academic Source",
            source_type="academic",
            publication_date="2022",
            geography="Comparable jurisdictions",
            page=4,
            text=f"Placeholder academic evidence on outcomes of similar policies to {topic}.",
            relevance_score=0.86,
            credibility_score=0.82,
        ),
        EvidenceItem(
            source_id="PLACEHOLDER-CASE",
            title=f"Case Study Relevant to: {topic}",
            organization="(placeholder) Case Study Source",
            source_type="case_study",
            publication_date="2021",
            geography="Comparable jurisdiction",
            page=2,
            text=f"Placeholder case-study evidence on implementing a policy like {topic}.",
            relevance_score=0.8,
            credibility_score=0.75,
        ),
    ]


def retrieve_policy_evidence(
    queries: List[str],
    geography: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    top_k: int = 6,
):
    """Retrieve ranked, deduplicated, citable policy evidence.

    Foundation behavior: in MOCK_MODE returns fixture EvidenceItems so the rest of
    the team is unblocked. The real path (shared `transportation` Chroma collection
    with metadata filtering, dedup, max-chunks-per-source, credibility ranking) is
    TODO for Person 2.
    """
    from config import MOCK_MODE, MAX_CHUNKS_PER_SOURCE

    if MOCK_MODE:
        items = _mock_evidence(queries, geography)
    else:
        # TODO(P2): real retrieval against the shared transportation collection:
        #   - embed queries, query collection with metadata `where` filters
        #     (geography, source_type), dedup by source_id, cap per source,
        #     rank by relevance * credibility, attach page numbers.
        raise NotImplementedError("Real policy retrieval not implemented yet (P2).")

    if source_types:
        items = [e for e in items if e.source_type in source_types]

    # Deduplicate + cap chunks per source, then rank by relevance * credibility.
    per_source: dict[str, int] = defaultdict(int)
    ranked = sorted(
        items, key=lambda e: e.relevance_score * e.credibility_score, reverse=True
    )
    out = []
    for e in ranked:
        if per_source[e.source_id] >= MAX_CHUNKS_PER_SOURCE:
            continue
        per_source[e.source_id] += 1
        out.append(e)
        if len(out) >= top_k:
            break
    logger.info("retrieve_policy_evidence returned %d items (mock=%s)", len(out), True)
    return out
