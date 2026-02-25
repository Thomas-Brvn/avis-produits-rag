import uuid
from pathlib import Path

import chromadb

import config
from src.embeddings import LocalEmbedder


class ReviewVectorStore:
    """ChromaDB-backed vector store for product reviews."""

    COLLECTION_NAME = "product_reviews"

    def __init__(
        self,
        persist_path: str = config.VECTOR_STORE_PATH,
        embedder: LocalEmbedder | None = None,
    ):
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or LocalEmbedder()
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[dict]) -> None:
        """Index a list of document dicts (text + metadata)."""
        if not documents:
            return
        texts = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedder.embed(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = config.MAX_RESULTS) -> list[dict]:
        """Return top-k matching documents for a query."""
        query_vec = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": text, "metadata": meta, "distance": dist})
        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
