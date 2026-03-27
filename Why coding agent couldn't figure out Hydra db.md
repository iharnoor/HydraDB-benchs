# Why Coding Agent Couldn't Figure Out HydraDB

> A root cause analysis of API documentation mismatches that caused Claude Code to misconfigure HydraDB endpoints, leading to failed ingestion and suboptimal retrieval.

**March 26, 2026 — HydraDB POC Benchmark Project**

---

## 1. Executive Summary

During a benchmark project comparing HydraDB vs ChromaDB vs Mem0, Claude Code (AI coding agent) repeatedly misconfigured HydraDB API calls. The root cause was documentation that doesn't match the actual API. Three specific discrepancies caused cascading failures:

### The Core Problem

The docs at `docs.hydradb.com/essentials/memories` show an API format that returns 404 when called. The actual working API uses different endpoints, field names, and payload structures.

| Documentation Says | API Actually Accepts | Impact |
|---|---|---|
| Endpoint `/memory/add` | `/memories/add_memory` | 404 Not Found |
| Text field `"content"` | `"text"` | 422 Validation Error |
| Structure: Flat object | `{"memories": [...]}` array | 422 "memories field required" |
| User ID required at top level | Optional | Minor |

---

## 2. The Documentation vs Reality

### What the docs show (`docs.hydradb.com/essentials/memories`)

```bash
# Documentation example — DOES NOT WORK
curl -X POST https://api.hydradb.com/memory/add \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "user_id": "john",
    "content": "John prefers dark mode and detailed technical explanations",
    "infer": true
  }'
```

### What actually works (verified by testing)

```bash
# Working API call — VERIFIED
curl -X POST https://api.hydradb.com/memories/add_memory \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "memories": [
      {
        "text": "John prefers dark mode and detailed technical explanations",
        "infer": true
      }
    ]
  }'
```

### Verification tests run

| Test | Endpoint | Payload Format | Result |
|---|---|---|---|
| Test 1 | `/memories/add_memory` | Flat with `content` + `infer` | 422 — "memories field required" |
| Test 2 | `/memories/add_memory` | Array with `content` + `infer` | 422 — "text or user_assistant_pairs required" |
| Test 3 | `/memories/add_memory` | Array with `text` + `infer` | 200 — "queued", infer: true |

---

## 3. Timeline of Errors

**Stage 1: Uploaded as knowledge (PDF), not memories**
Agent used `/ingestion/upload_knowledge` to upload relationship timeline as a PDF file. Knowledge and memories are different data types in HydraDB with different search endpoints.

**Stage 2: Searched via deprecated `/recall/qna`**
Since data was uploaded as knowledge, agent used `/recall/qna` (the knowledge search endpoint). This endpoint was deprecated long ago. No results returned.

**Stage 3: User flagged the wrong endpoint**
HydraDB team member pointed out: *"Use `/recall/recall_preferences`, the `/recall/qna` endpoint was deprecated long back."* Agent switched search endpoint but data was still uploaded as knowledge.

**Stage 4: Re-ingested as memories via `/memories/add_memory`**
Changed from `upload_knowledge(pdf)` to `add_memories([{"text": chunk}])`. Now `recall_preferences` found the data correctly.

**Stage 5: Tried to follow docs for `infer` flag**
Read docs at `docs.hydradb.com/essentials/memories`. Docs showed `/memory/add` with `"content"` field. Agent rewrote client to match docs. Got 404.

**Stage 6: Brute-force tested all three formats**
Tested flat vs array, `content` vs `text`, with and without `infer`. Found that original format with `text` field + `infer` inside array items works.

---

## 4. Root Cause Analysis

| Decision | Why It Was Wrong | Root Cause |
|---|---|---|
| Uploaded as knowledge (PDF) | The data was relationship memories/events, not reference documents. Memories are the right data type for personal timeline data. | Agent didn't understand memories vs knowledge taxonomy in HydraDB |
| Searched via `/recall/qna` | This was the matching endpoint for knowledge, but it was deprecated. | No deprecation warning returned by the API — silent failure |
| Trusted docs for `infer` flag format | Docs showed `/memory/add` with `content` — doesn't exist. | Documentation doesn't match the API |
| Treated HydraDB like a generic vector DB | Didn't learn its data model. | No OpenAPI spec or SDK with type hints to guide the agent |

*Source doc: `https://docs.hydradb.com/essentials/memories`*

---

## 5. The `infer` Flag — What It Actually Does

The `infer` flag is a per-memory ingestion parameter that controls how HydraDB processes data:

| `infer: true` | `infer: false` |
|---|---|
| Extracts entities (people, places, events) | Stores content as-is |
| Resolves pronouns via sliding window ("He" -> "Harnoor") | No pronoun resolution |
| Builds graph edges (`MENTIONED_IN`, `NEXT`, `CO_OCCURS`) | No graph construction |
| Creates inferred embeddings (separate from dense + BM25) | Only dense + BM25 embeddings |
| Extracts structured preferences | No preference extraction |

### Where `infer` goes in the payload

Inside each memory item in the array, **NOT** at the top level:

```json
{
  "tenant_id": "live_benchmark",
  "memories": [
    {
      "text": "Katie surprised Harnoor with a birthday celebration on December 23, 2020.",
      "infer": true  // <-- HERE, per memory item
    }
  ]
}
```

During recall, HydraDB uses a weighted hybrid search across all three embedding types: **dense + inferred + BM25**. Without `infer: true` at ingestion, the inferred embeddings and graph edges don't exist, and retrieval quality degrades.

---

## 6. Verified Working API Reference

### Add Memories

```
POST /memories/add_memory
```

```json
{
  "tenant_id": "your_tenant",
  "memories": [
    {"text": "memory content here", "infer": true},
    {"text": "another memory", "infer": true}
  ]
}
```

Response:

```json
{
  "success": true,
  "results": [
    {"source_id": "uuid", "status": "queued", "infer": true}
  ]
}
```

### Recall Preferences

```
POST /recall/recall_preferences
```

```json
{
  "tenant_id": "your_tenant",
  "query": "What car did they buy?",
  "max_results": 5
}
```

*Returns ranked chunks via hybrid search (dense + inferred + BM25)*

### Other Recall Endpoints

| Endpoint | Method | Use Case |
|---|---|---|
| `/recall/recall_preferences` | Hybrid | Memory/preference recall (primary) |
| `/recall/full_recall` | Hybrid | Full hybrid vector + graph + BM25 |
| `/recall/boolean_recall` | BM25 | Full-text keyword search |
| `/recall/qna` | Deprecated | **Do not use — removed** |

---

## 7. Recommendations for HydraDB Team

### Documentation Fixes Needed

These are UX/documentation issues that will affect every developer and AI agent integrating HydraDB:

1. **Fix the docs at `docs.hydradb.com/essentials/memories`**
   Update endpoint from `/memory/add` to `/memories/add_memory`, field from `content` to `text`, and structure from flat object to `memories` array.

2. **Return deprecation errors, not silent failures**
   If `/recall/qna` is deprecated, return a clear error message pointing users to `/recall/recall_preferences`. Don't silently return empty results.

3. **Document the memories vs knowledge data model**
   Prominently explain: *"Uploaded memories? Use `recall_preferences`. Uploaded knowledge files? Use `full_recall`."*

4. **Publish an OpenAPI spec**
   A machine-readable spec would let AI coding agents (and IDE autocomplete) discover the correct endpoints, field names, and types automatically.

5. **Surface the correct recall endpoint in ingestion responses**
   When a user uploads memories, the response could include: `"recall_endpoint": "/recall/recall_preferences"`

---

## 8. Lessons for AI Coding Agents

### What the agent should have done differently

These patterns would have caught the issue earlier:

- **Test before trusting docs.** Always make a test API call before building an entire client around documentation. Docs can be wrong.
- **Check for OpenAPI specs first.** Machine-readable specs are more reliable than prose documentation.
- **Ask the user about data model taxonomy.** "Is this memories or knowledge?" would have prevented the initial misclassification.
- **When an endpoint returns empty results, investigate.** Don't assume the data isn't there — check if you're using the wrong search endpoint.
