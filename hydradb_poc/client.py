"""
╔══════════════════════════════════════════════════════════════╗
║  HydraDB Client — Built from the real OpenAPI spec           ║
║                                                              ║
║  Endpoints (verified):                                       ║
║  POST /tenants/create         — create isolated memory space ║
║  POST /memories/add_memory    — ingest memory items          ║
║  POST /recall/full_recall     — hybrid vector+graph+BM25     ║
║  POST /recall/recall_preferences — memory/preference recall  ║
║  POST /recall/boolean_recall  — full-text BM25 search        ║
║  POST /ingestion/upload_knowledge — upload files              ║
║  POST /ingestion/verify_processing — check indexing status   ║
║  GET  /list/graph_relations_by_id — explore knowledge graph  ║
║                                                              ║
║  Base URL: https://api.hydradb.com                           ║
║  Auth: Bearer token                                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import requests
from dataclasses import dataclass, field


BASE_URL = "https://api.hydradb.com"


@dataclass
class HydraDBClient:
    api_key: str
    tenant_id: str = ""
    base_url: str = BASE_URL
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self):
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, json: dict | None = None, **kwargs) -> dict:
        resp = self._session.post(self._url(path), json=json, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._session.get(self._url(path), params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Tenant Management ────────────────────────────────────

    def create_tenant(self, tenant_id: str) -> dict:
        self.tenant_id = tenant_id
        return self._post("/tenants/create", {"tenant_id": tenant_id})

    def use_tenant(self, tenant_id: str):
        """Set tenant without creating (for existing tenants)."""
        self.tenant_id = tenant_id

    def list_tenants(self) -> dict:
        return self._get("/tenants/tenant_ids")

    # ── Memory Ingestion ─────────────────────────────────────

    def add_memories(self, memories: list[dict], sub_tenant_id: str = "", infer: bool = True) -> dict:
        """
        Add memory items via /memories/add_memory.
        Each item should have 'text' and optionally 'infer'.
        infer=True extracts preferences, entities, and graph edges.
        """
        for mem in memories:
            if "infer" not in mem:
                mem["infer"] = infer
        payload = {
            "memories": memories,
            "tenant_id": self.tenant_id,
        }
        if sub_tenant_id:
            payload["sub_tenant_id"] = sub_tenant_id
        return self._post("/memories/add_memory", payload)

    def add_memory(self, text: str, title: str = "", infer: bool = True) -> dict:
        """Add a single memory item."""
        item = {"text": text, "infer": infer}
        if title:
            item["title"] = title
        return self.add_memories([item])

    # ── File Upload ──────────────────────────────────────────

    def upload_knowledge(self, file_path: str) -> dict:
        """Upload a file via curl subprocess to avoid Python SSL issues with large multipart."""
        import subprocess, json as _json
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                self._url("/ingestion/upload_knowledge"),
                "-H", f"Authorization: Bearer {self.api_key}",
                "-F", f"files=@{file_path}",
                "-F", f"tenant_id={self.tenant_id}",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr}")
        return _json.loads(result.stdout)

    def verify_processing(self, file_ids: list[str]) -> dict:
        params = "&".join(f"file_ids={fid}" for fid in file_ids)
        return self._post(
            f"/ingestion/verify_processing?{params}&tenant_id={self.tenant_id}"
        )

    def wait_for_processing(self, file_ids: list[str], timeout: int = 120, interval: int = 5) -> bool:
        """Poll until all files are processed or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            result = self.verify_processing(file_ids)
            statuses = result.get("statuses", [])
            if all(s.get("indexing_status") == "completed" for s in statuses):
                return True
            time.sleep(interval)
        return False

    # ── Search & Recall ──────────────────────────────────────

    def full_recall(self, query: str, max_results: int = 10) -> dict:
        """
        Hybrid search: vector + graph + BM25.
        Returns raw ranked chunks without LLM generation.
        """
        return self._post("/recall/full_recall", {
            "tenant_id": self.tenant_id,
            "query": query,
            "max_results": max_results,
        })

    def recall_preferences(self, query: str, max_results: int = 10) -> dict:
        """
        Fast preference/memory recall.
        Uses hybrid search: dense + inferred + BM25.
        Inferred embeddings require infer=True during ingestion.
        """
        return self._post("/recall/recall_preferences", {
            "tenant_id": self.tenant_id,
            "query": query,
            "max_results": max_results,
        })

    def boolean_recall(self, query: str, max_results: int = 10, operator: str = "or") -> dict:
        """Full-text BM25 search."""
        return self._post("/recall/boolean_recall", {
            "tenant_id": self.tenant_id,
            "query": query,
            "max_results": max_results,
            "operator": operator,
        })

    # ── Graph ────────────────────────────────────────────────

    def graph_relations(self, source_id: str) -> dict:
        """Explore knowledge graph around a source."""
        return self._get("/list/graph_relations_by_id", {
            "tenant_id": self.tenant_id,
            "source_id": source_id,
        })

    # ── Content Listing ──────────────────────────────────────

    def list_content(self, kind: str = "all", page: int = 1, page_size: int = 20) -> dict:
        return self._post("/list/data", {
            "tenant_id": self.tenant_id,
            "kind": kind,
            "page": page,
            "page_size": page_size,
        })

    # ── Delete ───────────────────────────────────────────────

    def delete_memory(self, memory_id: str) -> dict:
        resp = self._session.delete(
            self._url(f"/memories/delete_memory?tenant_id={self.tenant_id}&memory_id={memory_id}"),
        )
        resp.raise_for_status()
        return resp.json()
