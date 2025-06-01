import chromadb
from logger import logger
from chromadb.config import Settings

class ChromaDBConnection:
    def __init__(self, path):
        logger.info("Creating ChromaDB connection...")
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

    def get_collection(self, name, metadata):
        logger.info(f"Accessing collection: {name}")
        collection = self.client.get_or_create_collection(name=name, metadata=metadata)
        logger.info(f"Collection size: {collection.count()}")
        return collection