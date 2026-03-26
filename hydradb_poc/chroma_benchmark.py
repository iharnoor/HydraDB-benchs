"""
╔══════════════════════════════════════════════════════════════════╗
║  BENCHMARK: HydraDB vs ChromaDB (Pure Vector DB)                 ║
║                                                                  ║
║  Same data. Same questions. Real API calls vs local vectors.     ║
║  ChromaDB = pure cosine similarity. No graph, no BM25, no hybrid.║
║                                                                  ║
║  Run: python3 -m hydradb_poc.chroma_benchmark                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv

from hydradb_poc.client import HydraDBClient
from hydradb_poc.chroma_client import ChromaClient
from hydradb_poc.ingest import TIMELINE_CHUNKS

load_dotenv()


# ═══════════════════════════════════════════════════════════════
#  TEST SCENARIOS — same as live_benchmark.py
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    category: str
    name: str
    question: str
    expected: str
    why_matters: str


TEST_CASES = [
    # ── Information Extraction ────────────────────────────
    TestCase(
        category="Information Extraction",
        name="Specific date recall",
        question="When did Harnoor and Katie meet for the second time?",
        expected="October 18, 2020",
        why_matters="Exact date retrieval from a specific memory",
    ),
    TestCase(
        category="Information Extraction",
        name="Buried detail in rich context",
        question="What car did Harnoor and Katie buy together?",
        expected="White Tesla Model 3",
        why_matters="Specific fact (car model + color) buried in descriptive paragraph",
    ),
    TestCase(
        category="Information Extraction",
        name="Cultural detail extraction",
        question="What did Katie wear for Navratri?",
        expected="Red and green lehenga with traditional jewelry",
        why_matters="Specific clothing detail from a cultural event description",
    ),

    # ── Multi-Session Reasoning ───────────────────────────
    TestCase(
        category="Multi-Session Reasoning",
        name="Synthesize relationship progression",
        question="How did Harnoor and Katie's relationship progress from casual dating to major commitments?",
        expected="First car date Nov 2020 → first formal dinner Valentine's 2021 → bought a Tesla together May 2022",
        why_matters="Requires combining 3+ memories to show relationship arc",
    ),
    TestCase(
        category="Multi-Session Reasoning",
        name="Cross-cultural exchange pattern",
        question="How did Harnoor and Katie embrace each other's cultures?",
        expected="Katie: Navratri/lehenga/Garba, wore Indian clothing at Taj Mahal. Harnoor: Thanksgiving with Katie's family, Christmas together.",
        why_matters="Must combine cultural exchange moments across multiple memories",
    ),

    # ── Temporal Reasoning ────────────────────────────────
    TestCase(
        category="Temporal Reasoning",
        name="Timeline ordering — counting repeated events",
        question="How many times did Harnoor and Katie go to Miami for New Year's?",
        expected="Twice — January 2021 and January 2022",
        why_matters="Must identify a repeated event across two different years",
    ),
    TestCase(
        category="Temporal Reasoning",
        name="First trip outside Florida",
        question="When was Harnoor and Katie's first trip outside of Florida?",
        expected="July 3, 2021 — Princeton University visit in New Jersey",
        why_matters="Must distinguish Florida trips from out-of-state trips",
    ),

    # ── Semantic Understanding ────────────────────────────
    TestCase(
        category="Semantic Understanding",
        name="Relationship milestone inference",
        question="What was the significance of Harnoor and Katie attending a wedding together?",
        expected="It signaled deeper commitment — being invited to important life events as a unit",
        why_matters="Must infer emotional/relational meaning, not just recall facts",
    ),
    TestCase(
        category="Semantic Understanding",
        name="Intercultural significance",
        question="Why was the Fourth of July celebration special for Harnoor?",
        expected="As an international student from India, experiencing this American tradition with Katie was culturally meaningful",
        why_matters="Must extract the cultural significance, not just 'they watched fireworks'",
    ),

    # ── Abstention / Precision ────────────────────────────
    TestCase(
        category="Abstention",
        name="Don't confuse similar events",
        question="When is Harnoor's birthday and how was it celebrated?",
        expected="December 23 — Katie surprised him with a blue gift bag at his apartment",
        why_matters="Must NOT confuse Harnoor's birthday with Katie's birthday",
    ),
    TestCase(
        category="Abstention",
        name="Don't hallucinate engagement",
        question="Are Harnoor and Katie engaged or married?",
        expected="Unknown / not mentioned — buying a car is the biggest commitment mentioned",
        why_matters="Should NOT infer engagement/marriage from buying a car together",
    ),

    # ── Negation (Vector DBs Can't "NOT") ─────────────────
    TestCase(
        category="Negation",
        name="Negation — not a birthday",
        question="What did Harnoor and Katie celebrate together that was NOT a birthday?",
        expected="Navratri, July 4th, Valentine's Day, Thanksgiving, Christmas, New Year's, wedding",
        why_matters="Vector search returns birthday chunks FIRST because 'birthday' is in the query — cosine similarity can't negate",
    ),
    TestCase(
        category="Negation",
        name="Negation — trips outside Florida",
        question="What trips did Harnoor and Katie take that were NOT in Florida?",
        expected="Princeton/NJ (July 2021), India (March 2022)",
        why_matters="Vector returns Miami and FSU trips (highest similarity to 'trip') even though they're IN Florida",
    ),

    # ── Temporal Adjacency ────────────────────────────────
    TestCase(
        category="Temporal Adjacency",
        name="Day-after reasoning",
        question="What did Harnoor and Katie do the day after visiting Princeton?",
        expected="July 4th fireworks celebration — they watched fireworks in a large crowd",
        why_matters="Vector search has no concept of 'the day after' — returns Princeton chunk or random travel chunks",
    ),
    TestCase(
        category="Temporal Adjacency",
        name="Chronological ordering of milestones",
        question="Put these events in chronological order: buying a car, visiting India, attending a wedding",
        expected="Wedding (Aug 2021) → India (March 2022) → Tesla (May 2022)",
        why_matters="Vector returns chunks in SIMILARITY order, not chronological — embeddings have no clock",
    ),

    # ── Entity Direction (WHO did WHAT to WHOM) ───────────
    TestCase(
        category="Entity Direction",
        name="Directional gift giving",
        question="What did Katie specifically do for Harnoor's birthday?",
        expected="Surprised him with a large blue 'Happy Birthday' gift bag at his apartment on December 23, 2020",
        why_matters="Vector returns both birthday chunks equally — can't distinguish Katie→Harnoor from Harnoor→Katie direction",
    ),
    TestCase(
        category="Entity Direction",
        name="One-directional cultural adoption",
        question="Which of Harnoor's Indian traditions did Katie participate in?",
        expected="Navratri (wore lehenga, danced Garba), wore Indian clothing at Taj Mahal",
        why_matters="Vector also returns Thanksgiving/Christmas (Harnoor adopting Katie's traditions) — can't filter by direction",
    ),

    # ── Geographic Entity Filtering ───────────────────────
    TestCase(
        category="Geographic Filtering",
        name="International vs domestic",
        question="Which events happened outside of the United States?",
        expected="Only the India trip — visiting the Taj Mahal in March 2022",
        why_matters="Vector returns any travel chunk (Miami, Princeton, India all score similarly on 'events outside') — no geographic entity awareness",
    ),

    # ── Aggregation Across All Chunks ─────────────────────
    TestCase(
        category="Aggregation",
        name="Exhaustive listing",
        question="List every holiday or celebration Harnoor and Katie shared together",
        expected="Harnoor's birthday, Katie's birthday, NYE 2021, NYE 2022, Valentine's Day, Navratri, July 4th, Thanksgiving, Christmas, wedding attendance",
        why_matters="Vector top-5 retrieval returns ~5 most 'celebration-like' chunks and misses the rest — can't aggregate across all 19 memories",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  RESULT TYPE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Result:
    system: str
    case_name: str
    category: str
    answer: str
    latency_ms: float
    chunks_returned: int = 0
    error: str = ""


# ═══════════════════════════════════════════════════════════════
#  CHROMADB RUNNER
# ═══════════════════════════════════════════════════════════════

def run_chroma() -> tuple[list[Result], ChromaClient]:
    """Ingest all timeline chunks into ChromaDB, then run queries."""
    print("\n  [ChromaDB] Initializing local vector database...")
    client = ChromaClient(collection_name="relationship_timeline")

    # Ingest all timeline chunks
    print(f"  [ChromaDB] Ingesting {len(TIMELINE_CHUNKS)} memory chunks...")
    client.add_memories(TIMELINE_CHUNKS)
    print(f"  [ChromaDB] {client.count()} chunks indexed (instant — local embeddings)")

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        start = time.time()
        try:
            hits = client.search(case.question, top_k=5)
            latency = (time.time() - start) * 1000

            if hits:
                answer = " | ".join(h["text"][:150] for h in hits[:3])
            else:
                answer = "(no results)"

            print(f"{latency:.1f}ms ({len(hits)} chunks)")
            results.append(Result(
                "ChromaDB", case.name, case.category,
                answer[:500], latency, len(hits),
            ))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result(
                "ChromaDB", case.name, case.category,
                "", latency, 0, str(e),
            ))

    return results, client


# ═══════════════════════════════════════════════════════════════
#  HYDRADB RUNNER
# ═══════════════════════════════════════════════════════════════

def run_hydradb(api_key: str) -> list[Result]:
    """Query HydraDB (data already ingested in live_benchmark tenant)."""
    client = HydraDBClient(api_key=api_key)
    client.use_tenant("live_benchmark")
    print(f"\n  [HydraDB] Using tenant: live_benchmark")

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        start = time.time()
        try:
            resp = client.recall_preferences(case.question, max_results=5)
            latency = (time.time() - start) * 1000

            chunks = resp.get("chunks", [])
            if chunks:
                answer = " | ".join(
                    c.get("chunk_content", "")[:150] for c in chunks[:3]
                )
            else:
                answer = str(resp)[:200]

            print(f"{latency:.0f}ms ({len(chunks)} chunks)")
            results.append(Result(
                "HydraDB", case.name, case.category,
                answer[:500], latency, len(chunks),
            ))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result(
                "HydraDB", case.name, case.category,
                "", latency, 0, str(e),
            ))

    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    hydra_key = os.getenv("HYDRADB_API_KEY", "")

    if not hydra_key:
        print("Set HYDRADB_API_KEY in .env")
        return

    print("=" * 70)
    print("  BENCHMARK: HydraDB vs ChromaDB (Pure Vector DB)")
    print(f"  {len(TEST_CASES)} test cases | 5 categories")
    print(f"  ChromaDB: local, pure cosine similarity, default embeddings")
    print(f"  HydraDB:  cloud, hybrid vector+graph+BM25")
    print("=" * 70)

    # ── Run ChromaDB ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING CHROMADB (Pure Vector)")
    print(f"{'─'*70}")
    chroma_results, chroma_client = run_chroma()

    # ── Run HydraDB ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING HYDRADB (Hybrid)")
    print(f"{'─'*70}")
    hydra_results = run_hydradb(hydra_key)

    # ── Side-by-Side ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS: SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    comparison = []
    for case, chroma, hydra in zip(TEST_CASES, chroma_results, hydra_results):
        print(f"\n  ┌─ [{case.category}] {case.name}")
        print(f"  │  Q: {case.question}")
        print(f"  │  Expected: {case.expected}")
        print(f"  ├─ ChromaDB ({chroma.latency_ms:.1f}ms, {chroma.chunks_returned} chunks): {chroma.answer[:100]}")
        print(f"  ├─ HydraDB  ({hydra.latency_ms:.0f}ms, {hydra.chunks_returned} chunks): {hydra.answer[:100]}")
        print(f"  └─ Why it matters: {case.why_matters}")

        comparison.append({
            "category": case.category,
            "name": case.name,
            "question": case.question,
            "expected": case.expected,
            "chroma_answer": chroma.answer,
            "chroma_latency_ms": round(chroma.latency_ms, 1),
            "chroma_chunks": chroma.chunks_returned,
            "chroma_error": chroma.error,
            "hydra_answer": hydra.answer,
            "hydra_latency_ms": round(hydra.latency_ms),
            "hydra_chunks": hydra.chunks_returned,
            "hydra_error": hydra.error,
            "why_matters": case.why_matters,
        })

    # ── Summary ───────────────────────────────────────────
    c_lats = [r.latency_ms for r in chroma_results]
    h_lats = [r.latency_ms for r in hydra_results]
    c_errs = sum(1 for r in chroma_results if r.error)
    h_errs = sum(1 for r in hydra_results if r.error)

    summary = {
        "chromadb": {
            "avg_latency_ms": round(sum(c_lats) / len(c_lats), 1),
            "p50_latency_ms": round(sorted(c_lats)[len(c_lats) // 2], 1),
            "min_latency_ms": round(min(c_lats), 1),
            "max_latency_ms": round(max(c_lats), 1),
            "errors": c_errs,
            "total": len(chroma_results),
            "type": "Pure vector (cosine similarity)",
            "embedding_model": "all-MiniLM-L6-v2 (Chroma default)",
            "location": "Local (in-memory)",
        },
        "hydradb": {
            "avg_latency_ms": round(sum(h_lats) / len(h_lats)),
            "p50_latency_ms": round(sorted(h_lats)[len(h_lats) // 2]),
            "min_latency_ms": round(min(h_lats)),
            "max_latency_ms": round(max(h_lats)),
            "errors": h_errs,
            "total": len(hydra_results),
            "type": "Hybrid (vector + graph + BM25)",
            "location": "Cloud API",
        },
    }

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'':25s} {'Avg':>10s} {'P50':>10s} {'Min':>10s} {'Max':>10s} {'Errors':>8s}")
    print(f"  {'ChromaDB (pure vector)':25s} {summary['chromadb']['avg_latency_ms']:>8.1f}ms {summary['chromadb']['p50_latency_ms']:>8.1f}ms {summary['chromadb']['min_latency_ms']:>8.1f}ms {summary['chromadb']['max_latency_ms']:>8.1f}ms {c_errs:>6d}/{len(chroma_results)}")
    print(f"  {'HydraDB (hybrid)':25s} {summary['hydradb']['avg_latency_ms']:>8d}ms {summary['hydradb']['p50_latency_ms']:>8d}ms {summary['hydradb']['min_latency_ms']:>8d}ms {summary['hydradb']['max_latency_ms']:>8d}ms {h_errs:>6d}/{len(hydra_results)}")

    print(f"\n  Note: ChromaDB runs locally (no network latency).")
    print(f"  The comparison focuses on RETRIEVAL QUALITY, not speed.")
    print(f"  Both systems got the same {len(TIMELINE_CHUNKS)} memory chunks.")

    # ── Save ──────────────────────────────────────────────
    output = {
        "summary": summary,
        "comparison": comparison,
        "data": {
            "total_chunks_ingested": len(TIMELINE_CHUNKS),
            "test_cases": len(TEST_CASES),
            "categories": list(set(c.category for c in TEST_CASES)),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = os.path.join(os.path.dirname(__file__), "chroma_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
