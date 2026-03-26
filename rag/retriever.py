"""
╔══════════════════════════════════════════════════════════╗
║  STEP 3: Retriever — Finding Relevant Context            ║
║                                                          ║
║  Query ──► Embed ──► Find similar chunks (vector search) ║
║                  ──► Expand via graph (follow edges)      ║
║                  ──► Return ranked context                ║
║                                                          ║
║  This is where GRAPH RAG shines over plain RAG:          ║
║  After finding the closest chunk by vector similarity,   ║
║  we TRAVERSE the graph to pull in connected context       ║
║  (neighboring chunks, co-occurring entities, etc.)       ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
from rag.graph_store import GraphStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_search(graph: GraphStore, query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """
    PART A: Vector similarity search.

    Compare the query embedding against every chunk's embedding
    and return the top-k most similar chunks.

    (In production, you'd use Neo4j's vector index for this.
     Here we do it in Python for clarity.)
    """
    with graph.driver.session() as session:
        result = session.run(
            "MATCH (c:Chunk) RETURN c.id as id, c.text as text, c.embedding as embedding"
        )
        chunks = []
        for record in result:
            score = cosine_similarity(query_embedding, record["embedding"])
            chunks.append({
                "id": record["id"],
                "text": record["text"],
                "score": score,
                "source": "vector_search",
            })

    # Sort by similarity (highest first)
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k]


def graph_expand(graph: GraphStore, chunk_ids: list[str]) -> list[dict]:
    """
    PART B: Graph expansion — the GRAPH in Graph RAG.

    Starting from the vector-matched chunks, traverse the graph:
    1. Follow NEXT edges → get surrounding context (document flow)
    2. Follow MENTIONED_IN ← edges → find entities in these chunks
    3. Follow CO_OCCURS edges → find related entities
    4. Follow MENTIONED_IN → edges → find OTHER chunks mentioning those entities

    This gives us contextually related chunks that pure vector
    search might miss!
    """
    with graph.driver.session() as session:
        result = session.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})
            OPTIONAL MATCH (c)-[:NEXT]-(neighbor:Chunk)
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
            OPTIONAL MATCH (e)-[:CO_OCCURS]-(related:Entity)
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(other:Chunk)
            WHERE other.id <> c.id
            WITH collect(DISTINCT {id: neighbor.id, text: neighbor.text}) as neighbors,
                 collect(DISTINCT {name: e.name, type: e.type}) as entities,
                 collect(DISTINCT {id: other.id, text: other.text}) as related_chunks
            RETURN neighbors, entities, related_chunks
            """,
            chunk_ids=chunk_ids,
        )
        record = result.single()
        if not record:
            return []

        expanded = []
        # Add neighboring chunks (from NEXT edges)
        for n in record["neighbors"]:
            if n["id"]:
                expanded.append({
                    "id": n["id"],
                    "text": n["text"],
                    "source": "graph_neighbor",
                })
        # Add chunks found through entity connections
        for rc in record["related_chunks"]:
            if rc["id"]:
                expanded.append({
                    "id": rc["id"],
                    "text": rc["text"],
                    "source": "graph_entity_hop",
                })

        # Deduplicate
        seen = set()
        unique = []
        for item in expanded:
            if item["id"] not in seen and item["id"] not in chunk_ids:
                seen.add(item["id"])
                unique.append(item)

        return unique


def retrieve(
    graph: GraphStore,
    query_embedding: list[float],
    top_k: int = 2,
) -> dict:
    """
    Full retrieval pipeline: vector search + graph expansion.

    Returns a dict with:
    - vector_results: chunks found by vector similarity
    - graph_results: additional chunks found via graph traversal
    - all_context: combined text for the LLM
    - traversal_log: step-by-step log for visualization
    """
    log = []

    # Step 1: Vector search
    vector_results = vector_search(graph, query_embedding, top_k=top_k)
    log.append({
        "step": "Vector Search",
        "detail": f"Found {len(vector_results)} similar chunks",
        "data": [{"id": r["id"], "score": round(r["score"], 3)} for r in vector_results],
    })

    # Step 2: Graph expansion
    chunk_ids = [r["id"] for r in vector_results]
    graph_results = graph_expand(graph, chunk_ids)
    log.append({
        "step": "Graph Expansion",
        "detail": f"Found {len(graph_results)} additional chunks via graph traversal",
        "data": [{"id": r["id"], "source": r["source"]} for r in graph_results],
    })

    # Combine context
    all_chunks = vector_results + graph_results
    all_context = "\n\n---\n\n".join(c["text"] for c in all_chunks)

    return {
        "vector_results": vector_results,
        "graph_results": graph_results,
        "all_context": all_context,
        "traversal_log": log,
    }
