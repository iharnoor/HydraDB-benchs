"""
Side-by-side benchmark: HydraDB vs Mem0
on the Harnoor & Katie relationship timeline.

Tests 5 memory capabilities:
  1. Information Extraction   - recall specific facts
  2. Multi-Fact Reasoning     - combine facts from different events
  3. Temporal Reasoning       - chronological ordering & time-range queries
  4. Knowledge Updates        - track evolving state across time
  5. Abstention               - know when info is not available

Run:
  python -m hydradb_poc.compare_benchmark          # offline (mock answers)
  python -m hydradb_poc.compare_benchmark --live    # live API calls
"""

import os
import json
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from hydradb_poc.client import HydraDBClient
from hydradb_poc.mem0_client import Mem0Client

load_dotenv()


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK QUESTIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Question:
    category: str
    question: str
    expected_answer: str
    why_hard: str  # why this is hard for a basic memory system


QUESTIONS = [
    # ── 1. Information Extraction ────────────────────────────
    Question(
        category="Information Extraction",
        question="When and where did Harnoor and Katie first meet?",
        expected_answer=(
            "They first connected in early October 2020 through mutual friends at "
            "Florida State University. Their second meeting was on October 18, 2020."
        ),
        why_hard="The 'first meeting' is referenced indirectly in the second meeting entry. "
        "A system must infer the first connection happened earlier in October 2020.",
    ),
    Question(
        category="Information Extraction",
        question="What kind of car did Harnoor and Katie buy together and when?",
        expected_answer="A white Tesla Model 3, purchased on May 7, 2022.",
        why_hard="Straightforward extraction, but the system must locate the specific entry "
        "among 18 events.",
    ),
    Question(
        category="Information Extraction",
        question="What was Katie wearing when they visited the Taj Mahal?",
        expected_answer="A beautiful turquoise traditional Indian salwar kameez.",
        why_hard="Requires finding the specific Taj Mahal entry and extracting a detail "
        "about clothing, not the main event.",
    ),

    # ── 2. Multi-Fact Reasoning ──────────────────────────────
    Question(
        category="Multi-Fact Reasoning",
        question="How many times did Katie wear traditional Indian clothing, and on what occasions?",
        expected_answer=(
            "Twice: (1) Navratri celebration on October 15, 2021 - she wore a red and green "
            "lehenga with dandiya sticks, and (2) at the Taj Mahal in March 2022 - she wore "
            "a turquoise salwar kameez."
        ),
        why_hard="Requires scanning across ALL entries and connecting two separate events "
        "where Katie wore Indian clothing. A vector search for 'Indian clothing' might "
        "only return one of the two.",
    ),
    Question(
        category="Multi-Fact Reasoning",
        question="Which holidays did Harnoor and Katie celebrate together in 2021?",
        expected_answer=(
            "New Year's (Jan 1 in Miami), Valentine's Day (Feb 14), Fourth of July (Jul 4), "
            "Navratri (Oct 15), Thanksgiving (Nov 25), and Christmas (Dec 25)."
        ),
        why_hard="Requires aggregating 6 separate entries filtered to 2021. A basic vector "
        "search for 'holidays 2021' may only surface 1-2 of the 6.",
    ),
    Question(
        category="Multi-Fact Reasoning",
        question="How did the cultural exchange progress in the relationship? Give examples from both sides.",
        expected_answer=(
            "It was mutual and progressive: Katie embraced Indian culture by celebrating Navratri "
            "in traditional attire (Oct 2021), visiting India and wearing Indian clothing at the "
            "Taj Mahal (Mar 2022). Harnoor embraced American culture by celebrating Thanksgiving "
            "with Katie's family (Nov 2021), Christmas (Dec 2021), Fourth of July fireworks (Jul 2021), "
            "and attending an American wedding (Aug 2021)."
        ),
        why_hard="Requires synthesizing events from both cultural perspectives across the "
        "entire timeline. No single chunk contains this cross-cultural summary.",
    ),

    # ── 3. Temporal Reasoning ────────────────────────────────
    Question(
        category="Temporal Reasoning",
        question="What events happened between July and December 2021?",
        expected_answer=(
            "July 3: Visited Princeton University. July 4: Fourth of July fireworks. "
            "August 21: First American wedding. October 15: Navratri celebration. "
            "November 25: Thanksgiving with Katie's family. December 25: Christmas together."
        ),
        why_hard="Requires filtering events by a specific 6-month time range. "
        "Vector similarity alone cannot filter by date range.",
    ),
    Question(
        category="Temporal Reasoning",
        question="What was the very first trip they took outside of Florida, and when?",
        expected_answer=(
            "Their first trip outside Florida was to Princeton, New Jersey on July 3, 2021, "
            "about eight months into the relationship."
        ),
        why_hard="Requires understanding geography (Miami is in Florida, Princeton is not) "
        "and temporal ordering to find the FIRST out-of-state trip.",
    ),
    Question(
        category="Temporal Reasoning",
        question="How long were they dating before they first traveled internationally together?",
        expected_answer=(
            "About a year and a half. They started dating in October 2020 and traveled to "
            "India in March 2022 (approximately 17 months)."
        ),
        why_hard="Requires calculating time duration between two events: relationship start "
        "and India trip. Also must know India trip is their first international travel.",
    ),
    Question(
        category="Temporal Reasoning",
        question="Put these events in chronological order: buying the Tesla, visiting the Taj Mahal, "
        "first Valentine's Day, Navratri celebration, meeting for the second time.",
        expected_answer=(
            "1. Meeting for the second time (Oct 18, 2020), "
            "2. First Valentine's Day (Feb 14, 2021), "
            "3. Navratri celebration (Oct 15, 2021), "
            "4. Visiting the Taj Mahal (Mar 2022), "
            "5. Buying the Tesla (May 7, 2022)."
        ),
        why_hard="Requires extracting dates from 5 different entries and sorting them. "
        "This is pure temporal reasoning that vector search cannot do.",
    ),

    # ── 4. Knowledge Updates / State Tracking ────────────────
    Question(
        category="Knowledge Updates",
        question="How did their transportation situation change over the course of the relationship?",
        expected_answer=(
            "Initially (Oct 2020), neither had a car and getting around Tallahassee required planning. "
            "By Nov 2020 they were using someone's car for dates (first car date, Katie driving). "
            "By Jan 2021 they were flying to Miami. By May 2022 they bought their own car together - "
            "a white Tesla Model 3."
        ),
        why_hard="Requires tracking the evolution of a single attribute (transportation) across "
        "multiple time points. A graph database with temporal edges handles this naturally.",
    ),
    Question(
        category="Knowledge Updates",
        question="How did the nature of their dates evolve from 2020 to 2022?",
        expected_answer=(
            "Started casual (lying on grass, car rides around town in late 2020), "
            "progressed to trips (Miami Jan 2021), formal dates (Valentine's dinner Feb 2021), "
            "then major milestones (wedding together Aug 2021, meeting family Nov 2021), "
            "international travel (India Mar 2022), and financial commitment (Tesla May 2022)."
        ),
        why_hard="Requires understanding the progression arc across the entire timeline, "
        "not just retrieving individual events.",
    ),

    # ── 5. Abstention ────────────────────────────────────────
    Question(
        category="Abstention",
        question="When did Harnoor and Katie get engaged?",
        expected_answer=(
            "There is no information about an engagement in the timeline. The document covers "
            "October 2020 through May 2022, and no engagement is mentioned."
        ),
        why_hard="The system must recognize that engagement is never mentioned and say so, "
        "rather than hallucinating an answer from wedding-related context.",
    ),
    Question(
        category="Abstention",
        question="What are the names of Harnoor's parents?",
        expected_answer=(
            "The document does not mention Harnoor's parents' names. It only says they met "
            "Harnoor's family during the India trip in March 2022."
        ),
        why_hard="The India trip entry mentions meeting family but never names them. "
        "A system might hallucinate names or guess based on cultural assumptions.",
    ),
    Question(
        category="Abstention",
        question="Where did Katie go to school before FSU?",
        expected_answer="This information is not available in the timeline.",
        why_hard="Katie's pre-FSU education is never mentioned. The system should not infer.",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    question: Question
    hydra_answer: str = ""
    hydra_latency_ms: float = 0
    hydra_sources_count: int = 0
    mem0_answer: str = ""
    mem0_latency_ms: float = 0
    mem0_sources_count: int = 0


def query_hydradb(client: HydraDBClient, question: str) -> tuple[str, float, int]:
    """Query HydraDB and return (answer, latency_ms, sources_count)."""
    start = time.time()
    try:
        result = client.recall_preferences(question, max_results=10)
        latency = (time.time() - start) * 1000
        chunks = result.get("chunks", result.get("sources", []))
        if chunks:
            answer = " | ".join(
                c.get("chunk_content", c.get("content", ""))[:200] for c in chunks[:5]
            )
        else:
            answer = str(result)
        sources = len(chunks)
        return answer, latency, sources
    except Exception as e:
        latency = (time.time() - start) * 1000
        return f"Error: {e}", latency, 0


def query_mem0(client: Mem0Client, question: str) -> tuple[str, float, int]:
    """
    Query Mem0 search and return (answer, latency_ms, sources_count).

    Mem0 search returns memories, not generated answers.
    We return the top memories as the "answer" for comparison.
    """
    start = time.time()
    try:
        result = client.search(question, top_k=10)
        latency = (time.time() - start) * 1000

        # Mem0 returns a list of memory results
        memories = result if isinstance(result, list) else result.get("results", result.get("memories", []))
        sources = len(memories)

        # Combine top memories into an answer-like string
        if memories:
            parts = []
            for m in memories[:5]:
                mem_text = m.get("memory", m.get("text", str(m)))
                score = m.get("score", "")
                score_str = f" (score: {score:.3f})" if isinstance(score, float) else ""
                parts.append(f"- {mem_text}{score_str}")
            answer = "\n".join(parts)
        else:
            answer = "No relevant memories found."

        return answer, latency, sources
    except Exception as e:
        latency = (time.time() - start) * 1000
        return f"Error: {e}", latency, 0


def run_live_benchmark(hydra: HydraDBClient | None, mem0: Mem0Client | None) -> list[BenchmarkResult]:
    """Run all questions against live APIs."""
    results = []

    for i, q in enumerate(QUESTIONS):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(QUESTIONS)}] {q.category}")
        print(f"  Q: {q.question}")
        print(f"{'='*70}")

        result = BenchmarkResult(question=q)

        # Query HydraDB
        if hydra:
            print("  [HydraDB] Querying...")
            result.hydra_answer, result.hydra_latency_ms, result.hydra_sources_count = \
                query_hydradb(hydra, q.question)
            print(f"  [HydraDB] {result.hydra_latency_ms:.0f}ms, {result.hydra_sources_count} sources")
            print(f"  [HydraDB] Answer: {result.hydra_answer[:200]}")

        # Query Mem0
        if mem0:
            print("  [Mem0] Querying...")
            result.mem0_answer, result.mem0_latency_ms, result.mem0_sources_count = \
                query_mem0(mem0, q.question)
            print(f"  [Mem0] {result.mem0_latency_ms:.0f}ms, {result.mem0_sources_count} sources")
            print(f"  [Mem0] Answer: {result.mem0_answer[:200]}")

        print(f"\n  Expected: {q.expected_answer[:200]}")
        results.append(result)

    return results


def save_results(results: list[BenchmarkResult], path: str):
    """Save benchmark results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(results),
        "categories": {},
        "results": [],
    }

    for r in results:
        cat = r.question.category
        if cat not in data["categories"]:
            data["categories"][cat] = {
                "count": 0,
                "hydra_avg_latency_ms": 0,
                "mem0_avg_latency_ms": 0,
            }
        data["categories"][cat]["count"] += 1
        data["categories"][cat]["hydra_avg_latency_ms"] += r.hydra_latency_ms
        data["categories"][cat]["mem0_avg_latency_ms"] += r.mem0_latency_ms

        data["results"].append({
            "category": r.question.category,
            "question": r.question.question,
            "expected": r.question.expected_answer,
            "why_hard": r.question.why_hard,
            "hydra_answer": r.hydra_answer,
            "hydra_latency_ms": round(r.hydra_latency_ms),
            "hydra_sources": r.hydra_sources_count,
            "mem0_answer": r.mem0_answer,
            "mem0_latency_ms": round(r.mem0_latency_ms),
            "mem0_sources": r.mem0_sources_count,
        })

    # Average latencies per category
    for cat in data["categories"]:
        n = data["categories"][cat]["count"]
        data["categories"][cat]["hydra_avg_latency_ms"] = round(
            data["categories"][cat]["hydra_avg_latency_ms"] / n
        )
        data["categories"][cat]["mem0_avg_latency_ms"] = round(
            data["categories"][cat]["mem0_avg_latency_ms"] / n
        )

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved to: {path}")
    return data


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY: HydraDB vs Mem0")
    print("=" * 80)

    categories = {}
    for r in results:
        cat = r.question.category
        if cat not in categories:
            categories[cat] = {"hydra_latency": [], "mem0_latency": [], "count": 0}
        categories[cat]["hydra_latency"].append(r.hydra_latency_ms)
        categories[cat]["mem0_latency"].append(r.mem0_latency_ms)
        categories[cat]["count"] += 1

    print(f"\n  {'Category':<25} {'Count':>5} {'HydraDB Avg':>12} {'Mem0 Avg':>12}")
    print(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*12}")
    for cat, data in categories.items():
        h_avg = sum(data["hydra_latency"]) / len(data["hydra_latency"])
        m_avg = sum(data["mem0_latency"]) / len(data["mem0_latency"])
        print(f"  {cat:<25} {data['count']:>5} {h_avg:>10.0f}ms {m_avg:>10.0f}ms")

    # Overall
    all_h = [r.hydra_latency_ms for r in results]
    all_m = [r.mem0_latency_ms for r in results]
    print(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*12}")
    print(f"  {'OVERALL':<25} {len(results):>5} {sum(all_h)/len(all_h):>10.0f}ms {sum(all_m)/len(all_m):>10.0f}ms")

    print("\n  Key differences:")
    print("  - HydraDB /recall_preferences returns recalled memories ranked by relevance")
    print("  - Mem0 /search returns raw memories ranked by similarity (no generation)")
    print("  - HydraDB uses vector-based retrieval for fast preference/memory recall")
    print("  - Mem0 uses vector similarity search on extracted memory facts")
    print()


def main():
    import sys
    live = "--live" in sys.argv

    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "compare_results.json")

    if live:
        hydradb_key = os.getenv("HYDRADB_API_KEY", "")
        mem0_key = os.getenv("MEM0_API_KEY", "")

        hydra = None
        mem0 = None

        if hydradb_key:
            hydra = HydraDBClient(api_key=hydradb_key)
            hydra.use_tenant("live_benchmark")
            print("  [HydraDB] Connected, using tenant: live_benchmark")
        else:
            print("  WARNING: HYDRADB_API_KEY not set")

        if mem0_key:
            mem0 = Mem0Client(api_key=mem0_key, user_id="relationship_benchmark")
            print("  [Mem0] Connected")
        else:
            print("  WARNING: MEM0_API_KEY not set")

        if not hydra and not mem0:
            print("  ERROR: No API keys set. Set HYDRADB_API_KEY and/or MEM0_API_KEY")
            sys.exit(1)

        results = run_live_benchmark(hydra, mem0)
        save_results(results, output_path)
        print_summary(results)
    else:
        print("\n  No --live flag. Generating question list for review.\n")
        for i, q in enumerate(QUESTIONS):
            print(f"  [{i+1}] [{q.category}]")
            print(f"      Q: {q.question}")
            print(f"      Expected: {q.expected_answer[:120]}...")
            print(f"      Why hard: {q.why_hard[:100]}...")
            print()

        print(f"  Total: {len(QUESTIONS)} questions across "
              f"{len(set(q.category for q in QUESTIONS))} categories")
        print(f"\n  To run live: python -m hydradb_poc.compare_benchmark --live")


if __name__ == "__main__":
    main()
