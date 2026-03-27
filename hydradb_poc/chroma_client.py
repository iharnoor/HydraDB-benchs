"""
Traditional Vector DB Client — Pure vector database baseline.

Uses an open-source embedding database with cosine similarity
for search. No graph, no BM25, no hybrid — just vectors.

This is the "traditional vector DB" baseline for benchmarks.
"""

import chromadb
from dataclasses import dataclass, field


@dataclass
class ChromaClient:
    collection_name: str = "benchmark"
    persist_dir: str = ""
    _client: chromadb.ClientAPI = field(default=None, repr=False)
    _collection: chromadb.Collection = field(default=None, repr=False)

    def __post_init__(self):
        # Ephemeral (in-memory) client — no persistence needed for benchmarks
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(self, text: str, doc_id: str = None, metadata: dict = None) -> str:
        """Add a single document. Returns the ID."""
        if doc_id is None:
            doc_id = f"doc_{self._collection.count()}"
        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata] if metadata else None,
        )
        return doc_id

    def add_memories(self, texts: list[str], metadata: dict = None) -> list[str]:
        """Add multiple documents at once."""
        start = self._collection.count()
        ids = [f"doc_{start + i}" for i in range(len(texts))]
        self._collection.add(
            documents=texts,
            ids=ids,
            metadatas=[metadata for _ in texts] if metadata else None,
        )
        return ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Pure vector similarity search.
        Returns list of {text, score, id}.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )
        items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            distances = results["distances"][0] if results.get("distances") else [0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
            for doc, dist, doc_id in zip(docs, distances, ids):
                items.append({
                    "text": doc,
                    "score": 1 - dist,  # Chroma returns distance, convert to similarity
                    "id": doc_id,
                })
        return items

    def count(self) -> int:
        return self._collection.count()
