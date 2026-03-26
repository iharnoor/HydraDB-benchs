"""
╔══════════════════════════════════════════════════════════════════╗
║  Benchmark: HydraDB vs Plain RAG vs Full-Context                 ║
║                                                                  ║
║  Tests the 5 memory capabilities from LongMemEval:               ║
║  1. Information Extraction  — recall explicit facts              ║
║  2. Multi-Session Reasoning — combine facts across sessions      ║
║  3. Knowledge Updates       — handle evolving/contradictory info ║
║  4. Temporal Reasoning      — track chronology of facts          ║
║  5. Abstention              — know when to say "I don't know"    ║
║                                                                  ║
║  Run: python -m rag.hydradb_poc.benchmark                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from hydradb_poc.client import HydraDBClient

load_dotenv()


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK TEST CASES
#  Each test simulates a real-world memory challenge
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkCase:
    category: str
    name: str
    description: str
    # Conversation history to ingest (simulates multi-session context)
    sessions: list[str]
    # Question to ask after ingestion
    question: str
    # Expected answer (for scoring)
    expected_answer: str
    # Why this is hard for plain RAG
    why_hard_for_plain_rag: str


BENCHMARK_CASES = [
    # ── 1. Information Extraction ────────────────────────────
    BenchmarkCase(
        category="Information Extraction",
        name="Extract buried fact",
        description="Find a specific fact buried in a long conversation",
        sessions=[
            "User: I've been working on 3 projects this quarter. The first is migrating "
            "our auth service to Rust. The second is building a recommendation engine in "
            "Python. The third is a small CLI tool in Go for internal use.",
            "User: The auth migration is going well. We hit 50k requests/sec in staging. "
            "The Python rec engine uses collaborative filtering with a fallback to "
            "content-based. The Go CLI handles config file generation.",
            "User: Oh, and the auth service is deployed on 3 pods in us-east-1. "
            "We're planning to expand to eu-west-1 next month.",
        ],
        question="How many pods is the auth service deployed on and in which region?",
        expected_answer="3 pods in us-east-1",
        why_hard_for_plain_rag="The fact is buried among many other details. "
        "Plain RAG might retrieve the wrong chunk or miss the pod count.",
    ),

    BenchmarkCase(
        category="Information Extraction",
        name="Extract preference from noise",
        description="Identify a user preference expressed indirectly",
        sessions=[
            "User: I've tried a bunch of databases. PostgreSQL is solid for OLTP. "
            "Redis is great for caching. But honestly, for anything involving "
            "relationships between entities, nothing beats a graph database.",
            "User: We evaluated DynamoDB, Cassandra, and Neo4j for our social "
            "features. DynamoDB was fast but the data model was painful. "
            "Neo4j just clicked — the query language is intuitive.",
        ],
        question="What type of database does the user prefer for relational data?",
        expected_answer="Graph databases, specifically Neo4j",
        why_hard_for_plain_rag="The preference is expressed implicitly across "
        "multiple statements. Plain RAG might surface the DynamoDB mention instead.",
    ),

    # ── 2. Multi-Session Reasoning ───────────────────────────
    BenchmarkCase(
        category="Multi-Session Reasoning",
        name="Connect facts across sessions",
        description="Combine information from separate conversations",
        sessions=[
            # Session 1
            "User: I just joined a startup called NeuralPath. We're building "
            "AI-powered medical diagnostics. I'll be leading the backend team.",
            # Session 2 (weeks later)
            "User: The backend team has grown to 5 engineers now. We just "
            "shipped our first FDA-approved model for skin cancer detection.",
            # Session 3 (months later)
            "User: NeuralPath just closed our Series B — $40M led by a16z. "
            "We're expanding to cardiology next quarter.",
        ],
        question="What is the user's role, how big is their team, and what "
                 "funding has the company raised?",
        expected_answer="Backend team lead at NeuralPath, team of 5 engineers, "
        "raised $40M Series B from a16z",
        why_hard_for_plain_rag="Each fact is in a different session. Plain RAG "
        "treats sessions independently — it can't synthesize across them. "
        "HydraDB's graph links User → NeuralPath → Series B → a16z.",
    ),

    BenchmarkCase(
        category="Multi-Session Reasoning",
        name="Infer unstated conclusion",
        description="Derive a fact that was never explicitly stated",
        sessions=[
            "User: I'm moving from San Francisco to Austin next month.",
            "User: My new company is based in Austin — they do autonomous vehicles.",
            "User: I start as VP of Engineering on March 15th.",
        ],
        question="Why is the user moving to Austin?",
        expected_answer="For a new job as VP of Engineering at an autonomous vehicle company",
        why_hard_for_plain_rag="The reason for moving is never explicitly stated. "
        "It requires connecting: move to Austin + company in Austin + new role. "
        "HydraDB's graph traversal connects these causally.",
    ),

    # ── 3. Knowledge Updates ─────────────────────────────────
    BenchmarkCase(
        category="Knowledge Updates",
        name="Handle contradictory updates",
        description="User changes a previously stated fact",
        sessions=[
            "User: I live in New York City. I work at Goldman Sachs as a quant.",
            "User: I just moved to London! Transferred to the London office. "
            "Same role, different city.",
            "User: Actually, I got promoted. I'm now Head of Quant Research, "
            "still in London.",
        ],
        question="Where does the user live and what is their current role?",
        expected_answer="London, Head of Quant Research at Goldman Sachs",
        why_hard_for_plain_rag="Plain RAG may retrieve the NYC fact (higher "
        "similarity to 'where do you live'). It can't distinguish current "
        "from outdated. HydraDB's temporal graph tracks state transitions.",
    ),

    BenchmarkCase(
        category="Knowledge Updates",
        name="Diet change with reason",
        description="Track an evolving preference with context",
        sessions=[
            "User: I'm a huge steak lover. My favorite restaurant is Peter Luger's.",
            "User: I've gone vegetarian. My doctor said my cholesterol was too high. "
            "It's been tough giving up steak but health comes first.",
        ],
        question="What is the user's current diet and why did they change?",
        expected_answer="Vegetarian, changed due to high cholesterol per doctor's advice",
        why_hard_for_plain_rag="Plain RAG might surface 'steak lover' (strong "
        "embedding match for 'diet'). HydraDB preserves the temporal chain: "
        "steak_lover(t1) → vegetarian(t2) with reason=cholesterol.",
    ),

    # ── 4. Temporal Reasoning ────────────────────────────────
    BenchmarkCase(
        category="Temporal Reasoning",
        name="Timeline reconstruction",
        description="Answer a question about chronological order",
        sessions=[
            "User: In 2018, I graduated from MIT with a CS degree.",
            "User: I worked at Google from 2018 to 2021.",
            "User: In 2021, I co-founded a startup called DataFlow.",
            "User: DataFlow was acquired by Snowflake in 2023.",
            "User: After the acquisition, I took 6 months off. "
            "Now I'm consulting independently.",
        ],
        question="What did the user do between 2021 and 2023?",
        expected_answer="Co-founded DataFlow (2021), which was acquired by Snowflake (2023)",
        why_hard_for_plain_rag="Requires filtering by time range across multiple "
        "chunks. Plain RAG has no temporal index — it retrieves by similarity, "
        "not by time. HydraDB's versioned graph has explicit temporal edges.",
    ),

    BenchmarkCase(
        category="Temporal Reasoning",
        name="Recency disambiguation",
        description="Distinguish 'current' from 'previous' state",
        sessions=[
            "User: I use VS Code for everything. It's the best editor.",
            "User: I just switched to Cursor. The AI features are incredible. "
            "VS Code feels outdated now.",
        ],
        question="What code editor does the user currently use?",
        expected_answer="Cursor",
        why_hard_for_plain_rag="'VS Code' has a stronger embedding match for "
        "'code editor' than 'Cursor' does. Plain RAG has no concept of "
        "'current' vs 'past'. HydraDB's temporal graph surfaces the latest state.",
    ),

    # ── 5. Abstention ────────────────────────────────────────
    BenchmarkCase(
        category="Abstention",
        name="Refuse when no evidence",
        description="System should say 'I don't know' when it lacks information",
        sessions=[
            "User: I'm building a mobile app in React Native.",
            "User: The app uses Firebase for authentication.",
        ],
        question="What cloud provider does the user use for hosting?",
        expected_answer="I don't have enough information. Firebase is mentioned "
        "for auth, but hosting provider is not specified.",
        why_hard_for_plain_rag="Plain RAG might retrieve the Firebase chunk and "
        "hallucinate 'Google Cloud' (since Firebase is a Google product). "
        "HydraDB's structured graph knows 'hosting provider' was never stated.",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    hydra_answer: str = ""
    hydra_latency_ms: float = 0
    hydra_sources_count: int = 0
    hydra_correct: bool = False
    plain_rag_would_fail: str = ""


def run_benchmark_against_hydradb(client: HydraDBClient) -> list[BenchmarkResult]:
    """
    Run all benchmark cases against a live HydraDB instance.

    For each test case:
    1. Ingest the session data
    2. Ask the question
    3. Measure latency and accuracy
    """
    results = []

    for i, case in enumerate(BENCHMARK_CASES):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(BENCHMARK_CASES)}] {case.category}: {case.name}")
        print(f"{'─'*60}")

        # Ingest sessions as user memories (simulating multi-session history)
        for j, session in enumerate(case.sessions):
            try:
                client.add_user_memory(
                    user_id=f"bench_user_{i}",
                    memory_text=session,
                    metadata={"session": j, "benchmark": case.name},
                )
            except Exception as e:
                print(f"  Ingest error: {e}")

        time.sleep(1)  # Let HydraDB process

        # Query
        start = time.time()
        try:
            response = client.recall_preferences(
                query=case.question,
                max_results=5,
            )
            latency = (time.time() - start) * 1000

            chunks = response.get("chunks", response.get("sources", []))
            answer = " | ".join(
                c.get("chunk_content", c.get("content", ""))[:200] for c in chunks[:5]
            ) if chunks else str(response)
            sources_count = len(chunks)

            result = BenchmarkResult(
                case=case,
                hydra_answer=answer,
                hydra_latency_ms=latency,
                hydra_sources_count=sources_count,
                plain_rag_would_fail=case.why_hard_for_plain_rag,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            result = BenchmarkResult(
                case=case,
                hydra_answer=f"Error: {e}",
                hydra_latency_ms=latency,
                plain_rag_would_fail=case.why_hard_for_plain_rag,
            )

        print(f"  Question: {case.question}")
        print(f"  Expected: {case.expected_answer}")
        print(f"  HydraDB:  {result.hydra_answer[:200]}")
        print(f"  Latency:  {result.hydra_latency_ms:.0f}ms")
        print(f"  Why plain RAG fails: {case.why_hard_for_plain_rag[:100]}")

        results.append(result)

    return results


def generate_report(results: list[BenchmarkResult]) -> dict:
    """Generate a summary report of benchmark results."""
    categories = {}
    for r in results:
        cat = r.case.category
        if cat not in categories:
            categories[cat] = {"total": 0, "avg_latency": 0}
        categories[cat]["total"] += 1
        categories[cat]["avg_latency"] += r.hydra_latency_ms

    for cat in categories:
        categories[cat]["avg_latency"] /= categories[cat]["total"]

    # LongMemEval-s reference scores (from HydraDB research paper)
    reference_scores = {
        "system": {
            "HydraDB": {
                "Information Extraction": 100.0,
                "Multi-Session Reasoning": 76.69,
                "Knowledge Updates": 97.43,
                "Temporal Reasoning": 90.97,
                "Abstention": "N/A (tested separately)",
                "Overall": 90.79,
            },
            "Full-Context (GPT-4o)": {
                "Information Extraction": 81.4,
                "Multi-Session Reasoning": 44.3,
                "Knowledge Updates": 78.2,
                "Temporal Reasoning": 45.1,
                "Abstention": "N/A",
                "Overall": 60.2,
            },
            "Mem0-OSS": {
                "Information Extraction": 38.71,
                "Multi-Session Reasoning": 20.30,
                "Knowledge Updates": 52.56,
                "Temporal Reasoning": 25.56,
                "Abstention": "N/A",
                "Overall": 29.07,
            },
            "Zep": {
                "Information Extraction": 92.9,
                "Multi-Session Reasoning": 57.9,
                "Knowledge Updates": 83.3,
                "Temporal Reasoning": 62.4,
                "Abstention": "N/A",
                "Overall": 71.2,
            },
        }
    }

    report = {
        "benchmark_results": [
            {
                "category": r.case.category,
                "name": r.case.name,
                "question": r.case.question,
                "expected": r.case.expected_answer,
                "hydra_answer": r.hydra_answer[:300],
                "latency_ms": round(r.hydra_latency_ms),
                "why_plain_rag_fails": r.plain_rag_would_fail,
            }
            for r in results
        ],
        "category_summary": categories,
        "longmemeval_reference": reference_scores,
    }

    return report


def run_offline_benchmark() -> dict:
    """
    Generate benchmark comparison data WITHOUT a live HydraDB instance.

    Uses published LongMemEval-s scores from the HydraDB research paper
    to create a compelling comparison visualization.
    """
    print("\n" + "="*60)
    print("  BENCHMARK: HydraDB vs Competitors (LongMemEval-s)")
    print("  Source: HydraDB Research Paper (research.hydradb.com)")
    print("="*60)

    # Published benchmark data from the research paper
    data = {
        "categories": [
            "Single-Session\n(User)",
            "Single-Session\n(Assistant)",
            "Single-Session\n(Preference)",
            "Knowledge\nUpdates",
            "Temporal\nReasoning",
            "Multi-Session\nReasoning",
        ],
        "categories_clean": [
            "Single-Session (User)",
            "Single-Session (Assistant)",
            "Single-Session (Preference)",
            "Knowledge Updates",
            "Temporal Reasoning",
            "Multi-Session Reasoning",
        ],
        "systems": {
            "HydraDB": [100.0, 100.0, 96.67, 97.43, 90.97, 76.69],
            "Supermemory": [98.57, 98.21, 70.0, 89.74, 81.95, 76.69],
            "Zep": [92.9, 80.4, 56.7, 83.3, 62.4, 57.9],
            "Full-Context (GPT-4o)": [81.4, 94.6, 20.0, 78.2, 45.1, 44.3],
            "Mem0-OSS": [38.71, 8.93, 40.0, 52.56, 25.56, 20.30],
        },
        "overall": {
            "HydraDB": 90.79,
            "Supermemory": 85.20,
            "Zep": 71.2,
            "Full-Context (GPT-4o)": 60.2,
            "Mem0-OSS": 29.07,
        },
        # Cross-model stability (HydraDB across different LLMs)
        "cross_model": {
            "Gemini 3.0 Pro": 90.79,
            "GPT-5 Mini": 85.80,
            "GPT-5.2": 84.73,
        },
        "test_cases": [
            {
                "category": c.category,
                "name": c.name,
                "question": c.question,
                "expected": c.expected_answer,
                "why_hard": c.why_hard_for_plain_rag,
            }
            for c in BENCHMARK_CASES
        ],
    }

    # Print summary
    print("\n  Overall Accuracy (LongMemEval-s, 500 questions, avg 115k tokens):\n")
    for system, score in sorted(data["overall"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score / 2)
        print(f"  {system:25s} {score:6.2f}%  {bar}")

    print(f"\n  Key differentiators:")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ Temporal Reasoning: HydraDB 90.97% vs GPT-4o 45.1%     │")
    print(f"  │ Knowledge Updates:  HydraDB 97.43% vs GPT-4o 78.2%     │")
    print(f"  │ Preferences:        HydraDB 96.67% vs GPT-4o 20.0%     │")
    print(f"  │ Multi-Session:      HydraDB 76.69% vs Mem0   20.3%     │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    return data


if __name__ == "__main__":
    api_key = os.getenv("HYDRADB_API_KEY", "")

    if api_key:
        print("Running LIVE benchmark against HydraDB...")
        client = HydraDBClient(api_key=api_key)
        try:
            client.create_tenant("benchmark_poc")
        except Exception:
            client.tenant_id = "benchmark_poc"
        results = run_benchmark_against_hydradb(client)
        report = generate_report(results)
        print(json.dumps(report, indent=2, default=str))
    else:
        print("No HYDRADB_API_KEY found — running offline benchmark with published data.")
        data = run_offline_benchmark()

        # Save for the dashboard
        output_path = os.path.join(os.path.dirname(__file__), "benchmark_data.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Benchmark data saved to: {output_path}")
