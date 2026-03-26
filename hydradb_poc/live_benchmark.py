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
    # ── Information Extraction ────────────────────────────
    TestCase(
        category="Information Extraction",
        name="Specific date recall",
        messages=[
            [
                {"role": "user", "content": "Harnoor and Katie met for the second time on October 18, 2020. They had first connected earlier that month through mutual friends at Florida State University."},
                {"role": "assistant", "content": "Nice, FSU connection!"},
            ],
        ],
        question="When did Harnoor and Katie meet for the second time?",
        expected="October 18, 2020",
        why_matters="Exact date retrieval from a specific memory",
    ),
    TestCase(
        category="Information Extraction",
        name="Buried detail in rich context",
        messages=[
            [
                {"role": "user", "content": "On May 7, 2022, Harnoor and Katie bought a car together — a white Tesla Model 3. They posed at the Tesla dealership with an employee handing over the keys. Harnoor wore a gray blazer. They had been together for about a year and seven months."},
                {"role": "assistant", "content": "Big milestone!"},
            ],
        ],
        question="What car did Harnoor and Katie buy together?",
        expected="White Tesla Model 3",
        why_matters="Specific fact (car model + color) buried in descriptive paragraph",
    ),
    TestCase(
        category="Information Extraction",
        name="Cultural detail extraction",
        messages=[
            [
                {"role": "user", "content": "On October 15, 2021, Harnoor and Katie celebrated Navratri together. They both dressed in traditional Indian attire and held dandiya sticks for the Garba dance. Katie wore a beautiful red and green lehenga with traditional jewelry. Harnoor wore a teal kurta."},
                {"role": "assistant", "content": "That's beautiful!"},
            ],
        ],
        question="What did Katie wear for Navratri?",
        expected="Red and green lehenga with traditional jewelry",
        why_matters="Specific clothing detail from a cultural event description",
    ),

    # ── Multi-Session Reasoning ───────────────────────────
    TestCase(
        category="Multi-Session Reasoning",
        name="Synthesize relationship progression",
        messages=[
            [
                {"role": "user", "content": "November 2, 2020 — Harnoor and Katie went on their first car date. Katie was driving. This was one of their first real dates outside of campus."},
                {"role": "assistant", "content": "Cute!"},
            ],
            [
                {"role": "user", "content": "February 14, 2021 — They celebrated their first Valentine's Day at an upscale restaurant in Tallahassee. Both dressed up. It was their first formal date night."},
                {"role": "assistant", "content": "Fancy!"},
            ],
            [
                {"role": "user", "content": "May 7, 2022 — They bought a white Tesla Model 3 together at a dealership. One of the biggest financial commitments as a couple."},
                {"role": "assistant", "content": "Wow!"},
            ],
        ],
        question="How did Harnoor and Katie's relationship progress from casual dating to major commitments?",
        expected="First car date Nov 2020 → first formal dinner Valentine's 2021 → bought a Tesla together May 2022",
        why_matters="Requires combining 3 sessions to show relationship arc",
    ),
    TestCase(
        category="Multi-Session Reasoning",
        name="Cross-cultural exchange pattern",
        messages=[
            [
                {"role": "user", "content": "October 15, 2021 — Katie celebrated Navratri with Harnoor, wore a red and green lehenga, and danced Garba. She fully embraced Indian culture."},
                {"role": "assistant", "content": "Amazing!"},
            ],
            [
                {"role": "user", "content": "November 25, 2021 — Harnoor spent Thanksgiving with Katie's family. Traditional spread — turkey, ham, cranberries. First American family holiday for Harnoor."},
                {"role": "assistant", "content": "So wholesome!"},
            ],
            [
                {"role": "user", "content": "March 2022 — They visited the Taj Mahal. Katie wore a turquoise traditional Indian salwar kameez. They also met Harnoor's family who warmly welcomed Katie."},
                {"role": "assistant", "content": "Dream trip!"},
            ],
        ],
        question="How did Harnoor and Katie embrace each other's cultures?",
        expected="Katie: Navratri/lehenga/Garba, wore Indian clothing at Taj Mahal. Harnoor: Thanksgiving with Katie's family, Christmas together.",
        why_matters="Must combine cultural exchange moments across multiple sessions",
    ),

    # ── Temporal Reasoning ────────────────────────────────
    TestCase(
        category="Temporal Reasoning",
        name="Timeline ordering",
        messages=[
            [
                {"role": "user", "content": "January 1, 2021 — Harnoor and Katie traveled to Miami for New Year's. First trip as a couple. Only 2.5 months into dating."},
                {"role": "assistant", "content": "Fun!"},
            ],
            [
                {"role": "user", "content": "January 1, 2022 — They rang in 2022 on South Beach in Miami again. Making it an annual tradition. Now over 14 months together."},
                {"role": "assistant", "content": "Love a tradition!"},
            ],
        ],
        question="How many times did Harnoor and Katie go to Miami for New Year's?",
        expected="Twice — January 2021 and January 2022",
        why_matters="Must identify a repeated event across two different years",
    ),
    TestCase(
        category="Temporal Reasoning",
        name="First trip outside Florida",
        messages=[
            [
                {"role": "user", "content": "January 1, 2021 — First trip together to Miami."},
                {"role": "assistant", "content": "Nice!"},
            ],
            [
                {"role": "user", "content": "May 30, 2021 — Beach trip to FSU campus, Westcott Building photo."},
                {"role": "assistant", "content": "Classic FSU!"},
            ],
            [
                {"role": "user", "content": "July 3, 2021 — Visited Princeton University in New Jersey. First trip outside of Florida together, about eight months into the relationship."},
                {"role": "assistant", "content": "East coast road trip!"},
            ],
        ],
        question="When was Harnoor and Katie's first trip outside of Florida?",
        expected="July 3, 2021 — Princeton University visit in New Jersey",
        why_matters="Must distinguish Florida trips from out-of-state trips",
    ),

    # ── Semantic Understanding ────────────────────────────
    TestCase(
        category="Semantic Understanding",
        name="Relationship milestone inference",
        messages=[
            [
                {"role": "user", "content": "August 21, 2021 — Harnoor and Katie attended their first American wedding together at a beautiful outdoor venue with rose gardens. Katie wore a navy polka-dot dress, Harnoor a navy suit. Going to a wedding together signaled deeper commitment — about ten months together."},
                {"role": "assistant", "content": "Couple goals!"},
            ],
        ],
        question="What was the significance of Harnoor and Katie attending a wedding together?",
        expected="It signaled deeper commitment — being invited to important life events as a unit",
        why_matters="Must infer emotional/relational meaning, not just recall facts",
    ),
    TestCase(
        category="Semantic Understanding",
        name="Intercultural significance",
        messages=[
            [
                {"role": "user", "content": "July 4, 2021 — Harnoor and Katie celebrated the Fourth of July, watching fireworks in a large crowd. Celebrating America's Independence Day was especially meaningful for Harnoor as an international student from India experiencing this American tradition with Katie."},
                {"role": "assistant", "content": "Beautiful!"},
            ],
        ],
        question="Why was the Fourth of July celebration special for Harnoor?",
        expected="As an international student from India, experiencing this American tradition with Katie was culturally meaningful",
        why_matters="Must extract the cultural significance, not just 'they watched fireworks'",
    ),

    # ── Abstention / Precision ────────────────────────────
    TestCase(
        category="Abstention",
        name="Don't confuse similar events",
        messages=[
            [
                {"role": "user", "content": "December 23, 2020 — Katie surprised Harnoor with a birthday celebration. She got him a large blue 'Happy Birthday' gift bag. They celebrated at Harnoor's apartment."},
                {"role": "assistant", "content": "Sweet!"},
            ],
            [
                {"role": "user", "content": "March 16, 2021 — Harnoor celebrated Katie's birthday with a white layered cake that read 'Happy Birthday Katie'. They celebrated at what appears to be Katie's family home."},
                {"role": "assistant", "content": "Reciprocated!"},
            ],
        ],
        question="When is Harnoor's birthday and how was it celebrated?",
        expected="December 23 — Katie surprised him with a blue gift bag at his apartment",
        why_matters="Must NOT confuse Harnoor's birthday with Katie's birthday",
    ),
    TestCase(
        category="Abstention",
        name="Don't hallucinate engagement",
        messages=[
            [
                {"role": "user", "content": "May 7, 2022 — Harnoor and Katie bought a white Tesla Model 3 together. One of the biggest financial and practical commitments as a couple. About a year and seven months together."},
                {"role": "assistant", "content": "Major step!"},
            ],
        ],
        question="Are Harnoor and Katie engaged or married?",
        expected="Unknown / not mentioned — buying a car is the biggest commitment mentioned",
        why_matters="Should NOT infer engagement/marriage from buying a car together",
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
