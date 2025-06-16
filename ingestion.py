import os
import tempfile
import multiprocessing as mp
from logger import logger
from db_connection import ChromaDBConnection
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DB_PATH, CHUNK_OVERLAP, CHUNK_SIZE, HNSW_SPACE, BATCH_SIZE, EMBEDDING_MODEL

def get_embeddings():
    logger.info("Loading embedding model...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return embeddings

def _loader_from_bytes(pdf_bytes: bytes) -> PyPDFLoader:
    """
    Try the in-memory constructor (langchain-community ≥ 0.0.15).
    Fall back to writing a temp file on older versions.
    """
    try:
        return PyPDFLoader.from_bytes(pdf_bytes)
    except AttributeError:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.flush()
        return PyPDFLoader(tmp.name) 

def process_documents(file_bytes_list, collection_name):
    """
    Process a list of (filename, pdf_bytes) tuples and index them in ChromaDB.
    
    Args:
        file_bytes_list: List of tuples (filename, pdf_bytes)
        collection_name: Name of the ChromaDB collection
    """
    logger.info(f"Indexing {len(file_bytes_list)} documents for collection: {collection_name}")
    
    if not file_bytes_list:
        logger.info(f"No documents to process for collection: {collection_name}")
        return
    
    total_batches = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    embedding_model = get_embeddings()

    db = ChromaDBConnection(DB_PATH)
    collection = db.get_collection(collection_name, {"hnsw:space": HNSW_SPACE})
    
    for filename, pdf_bytes in file_bytes_list:
        try:
            logger.info(f"Processing document: {filename}")
            
            # Validate that pdf_bytes is actually bytes
            if not isinstance(pdf_bytes, bytes):
                logger.error(f"Expected bytes for {filename}, got {type(pdf_bytes)}")
                continue
                
            loader = _loader_from_bytes(pdf_bytes)
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            
            logger.info(f"Extracted {len(chunks)} chunks from {filename}")

            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = os.path.basename(filename)
                documents.append(chunk.page_content)
                metadatas.append(chunk.metadata)
                ids.append(f"{os.path.basename(filename)}_{i}")

                if len(ids) >= BATCH_SIZE:
                    embeddings = [embedding_model.embed_query(d) for d in documents]
                    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                    documents, metadatas, ids = [], [], []
                    total_batches += 1
                    logger.info(f"Processed batch {total_batches} for {filename}")

            if documents:
                embeddings = [embedding_model.embed_query(d) for d in documents]
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                total_batches += 1
                logger.info(f"Processed final batch for {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            raise  # Re-raise to let the caller handle it

    logger.info(f"Finished processing {total_batches} batches for {collection_name}")