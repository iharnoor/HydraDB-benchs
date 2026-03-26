"""
╔══════════════════════════════════════════════════════════════════╗
║  HydraDB RAG Demo — Step-by-Step Walkthrough                     ║
║                                                                  ║
║  This demo shows exactly what HydraDB does differently           ║
║  from a plain RAG pipeline:                                      ║
║                                                                  ║
║  Plain RAG:    chunk → embed → cosine search → LLM               ║
║  HydraDB:      ingest → temporal graph + sliding window →        ║
║                multi-hop recall (vector+graph+sparse) →          ║
║                user/hive memory enrichment → LLM                 ║
║                                                                  ║
║  Run: python -m rag.hydradb_poc.demo                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from dotenv import load_dotenv

from hydradb_poc.client import HydraDBClient

load_dotenv()


def print_step(step_num: int, title: str, description: str):
    """Pretty-print a pipeline step."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}")
    print(f"  {description}")
    print()


def print_result(label: str, data):
    """Pretty-print a result."""
    print(f"  ✓ {label}:")
    if isinstance(data, dict):
        print(f"    {json.dumps(data, indent=4, default=str)[:500]}")
    else:
        print(f"    {data}")
    print()


def run_demo():
    """
    End-to-end HydraDB RAG demonstration.

    Shows each step of the pipeline with explanations.
    """
    api_key = os.getenv("HYDRADB_API_KEY", "")
    if not api_key:
        print("Set HYDRADB_API_KEY in your .env file")
        print("Get one at https://workbench.hydradb.com/")
        return

    client = HydraDBClient(api_key=api_key)

    # ── Step 1: Create a Memory Space (Tenant) ──────────────
    print_step(1, "CREATE MEMORY SPACE",
        "A Memory Space is an isolated data container.\n"
        "  No data crosses boundaries — like a separate database per customer.\n"
        "  This is how HydraDB handles multi-tenancy.")

    try:
        result = client.create_tenant("demo_rag_poc")
        print_result("Tenant created", result)
    except Exception as e:
        print(f"  (Tenant may already exist: {e})")
        client.tenant_id = "demo_rag_poc"

    # ── Step 2: Ingest Knowledge ─────────────────────────────
    print_step(2, "INGEST KNOWLEDGE",
        "Upload documents to HydraDB. Behind the scenes:\n"
        "  → Sliding Window Inference resolves pronouns & entities\n"
        "  → Git-style versioned graph captures relationships\n"
        "  → Multi-field embeddings (content + inferred + sparse)\n"
        "  This is NOT just 'chunk and embed' — it's full contextual understanding.")

    # Upload the sample knowledge base
    sample_path = os.path.join(os.path.dirname(__file__), "../sample_data/knowledge.txt")
    sample_path = os.path.normpath(sample_path)

    try:
        result = client.upload_knowledge(sample_path)
        print_result("Document uploaded", result)

        # Wait for processing
        print("  Waiting for HydraDB to process (sliding window + graph building)...")
        time.sleep(5)
    except Exception as e:
        print(f"  Upload result: {e}")

    # ── Step 3: Add User Memories ────────────────────────────
    print_step(3, "ADD USER MEMORIES",
        "User memories enable personalization across sessions.\n"
        "  Unlike plain RAG which is stateless, HydraDB remembers:\n"
        "  → User preferences, context, history\n"
        "  → These are automatically recalled when relevant")

    memories = [
        "User prefers graph databases over relational databases",
        "User is building a multi-agent system for document processing",
        "User's favorite programming language is Python",
        "User previously tried Pinecone but found it too limited for relational queries",
    ]

    for mem in memories:
        try:
            result = client.add_user_memory("demo_user", mem)
            print_result(f"Memory stored", mem)
        except Exception as e:
            print(f"  Memory result: {e}")

    # ── Step 4: Add Hive Memory ──────────────────────────────
    print_step(4, "ADD HIVE MEMORY (Shared Intelligence)",
        "Hive memory is shared across ALL agents in your system.\n"
        "  When Agent A learns something, Agent B automatically knows it.\n"
        "  This is unique to HydraDB — no other memory system does this.")

    hive_memories = [
        "Graph databases outperform vector-only stores for multi-hop reasoning tasks",
        "RAG systems need temporal awareness to handle knowledge updates correctly",
    ]

    for mem in hive_memories:
        try:
            result = client.add_hive_memory(mem)
            print_result("Hive memory stored", mem)
        except Exception as e:
            print(f"  Hive result: {e}")

    # ── Step 5: Query with Full Recall Pipeline ──────────────
    print_step(5, "QUERY — FULL RECALL PIPELINE",
        "Now we ask a question. HydraDB's recall pipeline:\n"
        "  1. Adaptive Query Expansion (reformulates your question N ways)\n"
        "  2. Weighted Hybrid Search (dense + inferred + BM25 sparse)\n"
        "  3. Graph-Augmented Retrieval (entity traversal)\n"
        "  4. Chunk-Level Graph Expansion (neighbor discovery)\n"
        "  5. Triple-Tier Reranking (vector + graph + entity fusion)\n"
        "  \n"
        "  Plain RAG just does: embed → cosine similarity → done.")

    questions = [
        "What is RAG and how does it reduce hallucination?",
        "How do graph databases compare to vector stores for connected data?",
        "What embedding model should I use for semantic search?",
    ]

    for q in questions:
        print(f"\n  Question: \"{q}\"")
        print(f"  {'-'*50}")
        try:
            result = client.recall_preferences(query=q, max_results=5)
            print_result("Recalled memories", result)
        except Exception as e:
            print(f"  Search result: {e}")

    # ── Step 6: Explore Graph Relations ──────────────────────
    print_step(6, "EXPLORE THE KNOWLEDGE GRAPH",
        "This is where HydraDB shines. We can traverse the graph\n"
        "  to see HOW concepts are connected — not just similarity scores.\n"
        "  \n"
        "  Plain RAG: 'These chunks are similar (cosine=0.87)'\n"
        "  HydraDB:   'RAG ──USES──► Vector Embeddings ──ENABLES──► Semantic Search'")

    entities_to_explore = ["RAG", "Neo4j", "Python"]

    for entity in entities_to_explore:
        print(f"\n  Exploring graph around: '{entity}'")
        try:
            result = client.graph_relations(entity)
            print_result(f"Graph for '{entity}'", result)
        except Exception as e:
            print(f"  Graph result: {e}")

    # ── Step 7: Recall User Memory ───────────────────────────
    print_step(7, "RECALL USER MEMORY",
        "Retrieve memories specific to this user.\n"
        "  HydraDB uses the same hybrid search for memories,\n"
        "  so it finds relevant memories even when phrased differently.")

    try:
        result = client.recall_user_memory(
            "demo_user",
            "What databases has the user tried?",
            max_results=5,
        )
        print_result("User memories recalled", result)
    except Exception as e:
        print(f"  Recall result: {e}")

    print(f"\n{'='*60}")
    print("  DEMO COMPLETE")
    print(f"{'='*60}")
    print("""
  What you just saw:
  ┌─────────────────────────────────────────────────────┐
  │  1. Created an isolated Memory Space (tenant)       │
  │  2. Ingested docs with sliding window inference     │
  │  3. Stored per-user memories (personalization)      │
  │  4. Stored hive memories (shared agent intelligence)│
  │  5. Queried with 5-stage recall pipeline            │
  │  6. Explored the knowledge graph (multi-hop)        │
  │  7. Recalled user-specific memories                 │
  └─────────────────────────────────────────────────────┘

  Plain RAG gives you steps 2 + half of 5.
  HydraDB gives you ALL of it, with 90.79% accuracy
  on LongMemEval-s (vs 60.2% for full-context baseline).
    """)


if __name__ == "__main__":
    run_demo()
