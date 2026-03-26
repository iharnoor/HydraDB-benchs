"""
Mem0 Platform Client for benchmark comparison.

Wraps the Mem0 Platform API (https://docs.mem0.ai).
Uses the REST API directly (no SDK dependency) to keep it
consistent with how HydraDBClient works.

Endpoints used:
  POST /v1/memories/   — add a memory
  POST /v1/memories/search/ — search memories
  GET  /v1/memories/   — list memories
  DELETE /v1/memories/:id/ — delete a memory
"""

import time
import requests
from dataclasses import dataclass, field


BASE_URL = "https://api.mem0.ai"


@dataclass
class Mem0Client:
    api_key: str
    user_id: str = "benchmark_user"
    base_url: str = BASE_URL
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self):
        self._session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, json: dict | None = None) -> dict:
        resp = self._session.post(self._url(path), json=json)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._session.get(self._url(path), params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = self._session.delete(self._url(path))
        resp.raise_for_status()
        return resp.json()

    # ── Add Memories ──────────────────────────────────────────

    def add_memory(self, text: str, metadata: dict | None = None) -> dict:
        """
        Add a single memory. Mem0 extracts facts automatically.
        """
        payload = {
            "messages": [
                {"role": "user", "content": text},
            ],
            "user_id": self.user_id,
        }
        if metadata:
            payload["metadata"] = metadata
        return self._post("/v1/memories/", payload)

    def add_memories_batch(self, texts: list[str], metadata: dict | None = None) -> list[dict]:
        """Add multiple memories sequentially (Mem0 has no batch endpoint)."""
        results = []
        for text in texts:
            result = self.add_memory(text, metadata=metadata)
            results.append(result)
        return results

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> dict:
        """
        Search memories by semantic similarity.
        Returns ranked memories with scores.
        """
        return self._post("/v1/memories/search/", {
            "query": query,
            "user_id": self.user_id,
            "top_k": top_k,
        })

    # ── List & Delete ─────────────────────────────────────────

    def list_memories(self, page: int = 1, page_size: int = 50) -> dict:
        """List all memories for the user."""
        return self._get("/v1/memories/", {
            "user_id": self.user_id,
            "page": page,
            "page_size": page_size,
        })

    def delete_memory(self, memory_id: str) -> dict:
        """Delete a specific memory."""
        return self._delete(f"/v1/memories/{memory_id}/")

    def delete_all(self) -> dict:
        """Delete all memories for the user."""
        return self._delete(f"/v1/memories/")
