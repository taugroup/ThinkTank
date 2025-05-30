from collections import defaultdict

from logger import logger
from db_connection import ChromaDBConnection
from typing import List, Tuple
from config import DB_PATH, HNSW_SPACE
from ingestion import get_embeddings

def retrieve_documents(queries: List[str], collection_name: str) -> List[str]:
    """
    Performs vector search on collections with dense retrieval.

    Args:
        queries (List[str]): List of translated user queries.
        collection_name (str): Name of the collection in ChromaDB.

    Returns:
        List (List[Tuple[str, str, str]]): List of (document_id, document_text, source_metadata).
    """
    logger.info("Performing retrieval...")
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
    dense_results = [ doc_text for _, doc_text, _ in zip(top_ids, docs["documents"], docs["metadatas"])]
    logger.info(f"Retrieved {dense_results} documents from collection '{collection_name}'.")
    return dense_results