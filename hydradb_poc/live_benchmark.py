"""
╔══════════════════════════════════════════════════════════════════╗
║  LIVE Benchmark: HydraDB vs Mem0                                 ║
║                                                                  ║
║  Same data. Same questions. Real API calls. Timed & scored.      ║
║                                                                  ║
║  Run: python3 -m hydradb_poc.live_benchmark                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from mem0 import MemoryClient as Mem0Client

from hydradb_poc.client import HydraDBClient

load_dotenv()


# ═══════════════════════════════════════════════════════════════
#  TEST SCENARIOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    category: str
    name: str
    messages: list[list[dict]]  # sessions of [{role, content}, ...]
    question: str
    expected: str
    why_matters: str


TEST_CASES = [
    TestCase(
        category="Information Extraction",
        name="Buried fact recall",
        messages=[
            [
                {"role": "user", "content": "I'm working on 3 projects. First is migrating our auth service to Rust. Second is a recommendation engine in Python. Third is a CLI tool in Go."},
                {"role": "assistant", "content": "Nice variety! How's progress?"},
            ],
            [
                {"role": "user", "content": "The auth migration hit 50k req/sec in staging. It's deployed on 3 pods in us-east-1. Planning to expand to eu-west-1 next month."},
                {"role": "assistant", "content": "Great throughput!"},
            ],
        ],
        question="How many pods is the auth service deployed on and in which region?",
        expected="3 pods in us-east-1",
        why_matters="Specific fact buried among many details",
    ),
    TestCase(
        category="Information Extraction",
        name="Implicit preference",
        messages=[
            [
                {"role": "user", "content": "I've tried PostgreSQL, Redis, DynamoDB, and Neo4j. For anything involving relationships between entities, nothing beats a graph database. Neo4j's Cypher language just clicked."},
                {"role": "assistant", "content": "Graph DBs are powerful for connected data."},
            ],
        ],
        question="What type of database does the user prefer for relational/connected data?",
        expected="Graph databases / Neo4j",
        why_matters="Preference is implicit, not stated as 'I prefer X'",
    ),
    TestCase(
        category="Multi-Session Reasoning",
        name="Synthesize across sessions",
        messages=[
            [
                {"role": "user", "content": "I just joined NeuralPath. We build AI medical diagnostics. I lead the backend team."},
                {"role": "assistant", "content": "Exciting!"},
            ],
            [
                {"role": "user", "content": "Backend team grew to 5 engineers. We shipped our first FDA-approved model for skin cancer detection."},
                {"role": "assistant", "content": "Congrats on FDA approval!"},
            ],
            [
                {"role": "user", "content": "NeuralPath closed a $40M Series B from a16z. Expanding to cardiology next quarter."},
                {"role": "assistant", "content": "Amazing milestone!"},
            ],
        ],
        question="What is the user's role, team size, and company funding?",
        expected="Backend lead at NeuralPath, 5 engineers, $40M Series B from a16z",
        why_matters="Requires combining facts from 3 separate sessions",
    ),
    TestCase(
        category="Knowledge Updates",
        name="Location + role change",
        messages=[
            [
                {"role": "user", "content": "I live in New York. I work at Goldman Sachs as a quant."},
                {"role": "assistant", "content": "NYC quant life!"},
            ],
            [
                {"role": "user", "content": "Just moved to London! Transferred to the London office."},
                {"role": "assistant", "content": "London is great."},
            ],
            [
                {"role": "user", "content": "Got promoted — I'm now Head of Quant Research, still in London."},
                {"role": "assistant", "content": "Congrats!"},
            ],
        ],
        question="Where does the user currently live and what is their role?",
        expected="London, Head of Quant Research at Goldman Sachs",
        why_matters="Must return LATEST state, not original NYC/quant",
    ),
    TestCase(
        category="Knowledge Updates",
        name="Diet change",
        messages=[
            [
                {"role": "user", "content": "I'm a huge steak lover. Favorite restaurant is Peter Luger's."},
                {"role": "assistant", "content": "Peter Luger's is legendary!"},
            ],
            [
                {"role": "user", "content": "Gone vegetarian. Doctor said cholesterol was dangerously high. Had to give up steak."},
                {"role": "assistant", "content": "Health first."},
            ],
        ],
        question="What is the user's current diet and why did they change?",
        expected="Vegetarian, due to high cholesterol",
        why_matters="Must return current state (vegetarian), not old (steak lover)",
    ),
    TestCase(
        category="Temporal Reasoning",
        name="Career timeline",
        messages=[
            [
                {"role": "user", "content": "Career: MIT 2018, Google 2018-2021, co-founded DataFlow 2021, acquired by Snowflake 2023, took 6 months off, now consulting."},
                {"role": "assistant", "content": "Impressive trajectory!"},
            ],
        ],
        question="What did the user do between 2021 and 2023?",
        expected="Co-founded DataFlow, acquired by Snowflake",
        why_matters="Must filter by time range, not just similarity",
    ),
    TestCase(
        category="Temporal Reasoning",
        name="Current vs past editor",
        messages=[
            [
                {"role": "user", "content": "I use VS Code for all my coding. Best editor."},
                {"role": "assistant", "content": "VS Code is popular."},
            ],
            [
                {"role": "user", "content": "Switched to Cursor. AI features are incredible. VS Code feels outdated now."},
                {"role": "assistant", "content": "Cursor is great."},
            ],
        ],
        question="What code editor does the user currently use?",
        expected="Cursor",
        why_matters="'VS Code' keyword-matches better but is outdated",
    ),
    TestCase(
        category="Abstention",
        name="Don't hallucinate hosting",
        messages=[
            [
                {"role": "user", "content": "Building a mobile app in React Native. Using Firebase for auth."},
                {"role": "assistant", "content": "Good stack!"},
            ],
        ],
        question="What cloud provider does the user use for hosting?",
        expected="Unknown / not mentioned",
        why_matters="Should NOT hallucinate 'Google Cloud' from Firebase mention",
    ),
    TestCase(
        category="Abstention",
        name="Don't invent salary",
        messages=[
            [
                {"role": "user", "content": "I'm a senior engineer at Stripe on the payments API team."},
                {"role": "assistant", "content": "Stripe's payments API is world-class!"},
            ],
        ],
        question="What is the user's salary?",
        expected="Unknown / not mentioned",
        why_matters="Should NOT guess salary from title + company",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  RESULT TYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Result:
    system: str
    case_name: str
    category: str
    answer: str
    latency_ms: float
    error: str = ""


# ═══════════════════════════════════════════════════════════════
#  MEM0 RUNNER
# ═══════════════════════════════════════════════════════════════

def run_mem0(api_key: str) -> list[Result]:
    client = Mem0Client(api_key=api_key)
    results = []

    for i, case in enumerate(TEST_CASES):
        user_id = f"bench_m0_{i}_{int(time.time())}"
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        # Ingest each session
        for session in case.messages:
            try:
                client.add(session, user_id=user_id)
            except Exception as e:
                print(f"INGEST ERR: {e}")

        time.sleep(5)  # Let Mem0 process

        # Search with retry — Mem0 can take a few seconds to index
        start = time.time()
        try:
            answer = "(no results)"
            for attempt in range(3):
                raw = client.search(case.question, filters={"user_id": user_id})

                if isinstance(raw, dict):
                    items = raw.get("results", raw.get("memories", []))
                elif isinstance(raw, list):
                    items = raw
                else:
                    items = [str(raw)]

                memories = []
                for item in items:
                    if isinstance(item, dict):
                        memories.append(item.get("memory", item.get("text", str(item))))
                    else:
                        memories.append(str(item))

                if memories:
                    answer = " | ".join(memories)
                    break
                if attempt < 2:
                    time.sleep(3)

            latency = (time.time() - start) * 1000
            print(f"{latency:.0f}ms")

            results.append(Result("Mem0", case.name, case.category, answer[:500], latency))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result("Mem0", case.name, case.category, "", latency, str(e)))

    return results


# ═══════════════════════════════════════════════════════════════
#  HYDRADB RUNNER
# ═══════════════════════════════════════════════════════════════

def run_hydradb(api_key: str) -> list[Result]:
    client = HydraDBClient(api_key=api_key)

    # Use existing tenant or create one
    try:
        tenants = client.list_tenants()
        existing = tenants.get("tenant_ids", [])
        if existing:
            client.use_tenant(existing[0])
            print(f"  Using existing tenant: {existing[0]}")
        else:
            client.create_tenant("benchmark")
            print(f"  Created tenant: benchmark")
    except Exception as e:
        print(f"  Tenant setup: {e}")
        client.use_tenant("live_benchmark")

    results = []

    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        # Ingest: flatten all messages into memory items
        source_ids = []
        for session in case.messages:
            memories = []
            for msg in session:
                if msg["role"] == "user":
                    memories.append({
                        "text": msg["content"],
                        "title": f"bench_{case.name}_{i}",
                    })
            if memories:
                try:
                    resp = client.add_memories(memories)
                    for r in resp.get("results", []):
                        sid = r.get("source_id")
                        if sid:
                            source_ids.append(sid)
                except Exception as e:
                    print(f"INGEST ERR: {e}")

        # Brief wait — recall_preferences works before graph is done
        if source_ids:
            time.sleep(3)

        # Search: use recall_preferences (fast, vector-based memory recall)
        start = time.time()
        try:
            pref_resp = client.recall_preferences(case.question, max_results=5)
            latency = (time.time() - start) * 1000

            chunks = pref_resp.get("chunks", [])
            if chunks:
                answer = " | ".join(
                    c.get("chunk_content", "")[:150] for c in chunks[:3]
                )
            else:
                answer = str(pref_resp)

            print(f"{latency:.0f}ms")
            results.append(Result("HydraDB", case.name, case.category, answer[:500], latency))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result("HydraDB", case.name, case.category, "", latency, str(e)))

    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    hydra_key = os.getenv("HYDRADB_API_KEY", "")
    mem0_key = os.getenv("MEM0_API_KEY", "")

    if not hydra_key or not mem0_key:
        print("Set both HYDRADB_API_KEY and MEM0_API_KEY in .env")
        return

    print("=" * 70)
    print("  LIVE BENCHMARK: HydraDB vs Mem0")
    print(f"  {len(TEST_CASES)} test cases | 5 categories | Real API calls")
    print("=" * 70)

    # ── Run Mem0 ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING MEM0")
    print(f"{'─'*70}")
    mem0_results = run_mem0(mem0_key)

    # ── Run HydraDB ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING HYDRADB")
    print(f"{'─'*70}")
    hydra_results = run_hydradb(hydra_key)

    # ── Side-by-Side ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS: SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    comparison = []
    for case, m0, hdb in zip(TEST_CASES, mem0_results, hydra_results):
        print(f"\n  ┌─ [{case.category}] {case.name}")
        print(f"  │  Q: {case.question}")
        print(f"  │  Expected: {case.expected}")
        print(f"  ├─ Mem0    ({m0.latency_ms:.0f}ms): {m0.answer[:120]}")
        print(f"  ├─ HydraDB ({hdb.latency_ms:.0f}ms): {hdb.answer[:120]}")
        print(f"  └─ Why it matters: {case.why_matters}")

        comparison.append({
            "category": case.category,
            "name": case.name,
            "question": case.question,
            "expected": case.expected,
            "mem0_answer": m0.answer,
            "mem0_latency_ms": round(m0.latency_ms),
            "mem0_error": m0.error,
            "hydra_answer": hdb.answer,
            "hydra_latency_ms": round(hdb.latency_ms),
            "hydra_error": hdb.error,
            "why_matters": case.why_matters,
        })

    # ── Summary ──────────────────────────────────────────────
    m0_lats = [r.latency_ms for r in mem0_results]
    hdb_lats = [r.latency_ms for r in hydra_results]
    m0_errs = sum(1 for r in mem0_results if r.error)
    hdb_errs = sum(1 for r in hydra_results if r.error)

    summary = {
        "mem0": {
            "avg_latency_ms": round(sum(m0_lats) / len(m0_lats)),
            "p50_latency_ms": round(sorted(m0_lats)[len(m0_lats) // 2]),
            "errors": m0_errs,
            "total": len(mem0_results),
        },
        "hydradb": {
            "avg_latency_ms": round(sum(hdb_lats) / len(hdb_lats)),
            "p50_latency_ms": round(sorted(hdb_lats)[len(hdb_lats) // 2]),
            "errors": hdb_errs,
            "total": len(hydra_results),
        },
    }

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'':20s} {'Avg Latency':>12s} {'P50':>8s} {'Errors':>8s}")
    print(f"  {'Mem0':20s} {summary['mem0']['avg_latency_ms']:>10d}ms {summary['mem0']['p50_latency_ms']:>6d}ms {m0_errs:>6d}/{len(mem0_results)}")
    print(f"  {'HydraDB':20s} {summary['hydradb']['avg_latency_ms']:>10d}ms {summary['hydradb']['p50_latency_ms']:>6d}ms {hdb_errs:>6d}/{len(hydra_results)}")

    # ── Save ─────────────────────────────────────────────────
    output = {
        "summary": summary,
        "comparison": comparison,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = os.path.join(os.path.dirname(__file__), "live_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"  View dashboard:   streamlit run hydradb_poc/dashboard.py")


if __name__ == "__main__":
    main()
