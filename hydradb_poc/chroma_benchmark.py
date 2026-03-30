"""
╔══════════════════════════════════════════════════════════════════╗
║  BENCHMARK: HydraDB vs Traditional Vector DB                      ║
║                                                                  ║
║  Same data. Same questions. Real API calls vs local vectors.     ║
║  Vector DB = pure cosine similarity. No graph, no BM25, no hybrid.║
║                                                                  ║
║  Run: python3 -m hydradb_poc.chroma_benchmark                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
import fitz  # PyMuPDF
from google import genai

from hydradb_poc.client import HydraDBClient
from hydradb_poc.chroma_client import ChromaClient

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
    # gold_keywords: a chunk is a "hit" if ALL keywords in at least one
    # gold group appear in it.  Groups are OR'd, keywords within a group
    # are AND'd.  e.g. [["Princeton", "New Jersey"], ["Taj Mahal"]]
    # means a chunk is relevant if it mentions (Princeton AND New Jersey)
    # OR (Taj Mahal).
    gold_keywords: list[list[str]] = None  # set via field default_factory below

    def __post_init__(self):
        if self.gold_keywords is None:
            self.gold_keywords = []


TEST_CASES = [
    # ── Information Extraction ────────────────────────────
    TestCase(
        category="Information Extraction",
        name="Specific date recall",
        question="When did Harnoor and Katie meet for the second time?",
        expected="October 18, 2020",
        why_matters="Exact date retrieval from a specific memory",
        gold_keywords=[["October 18, 2020", "second time"]],
    ),
    TestCase(
        category="Information Extraction",
        name="Buried detail in rich context",
        question="What car did Harnoor and Katie buy together?",
        expected="White Tesla Model 3",
        why_matters="Specific fact (car model + color) buried in descriptive paragraph",
        gold_keywords=[["Tesla Model 3"]],
    ),
    TestCase(
        category="Information Extraction",
        name="Cultural detail extraction",
        question="What did Katie wear for Navratri?",
        expected="Red and green lehenga with traditional jewelry",
        why_matters="Specific clothing detail from a cultural event description",
        gold_keywords=[["lehenga", "Navratri"]],
    ),

    # ── Multi-Session Reasoning ───────────────────────────
    TestCase(
        category="Multi-Session Reasoning",
        name="Synthesize relationship progression",
        question="How did Harnoor and Katie's relationship progress from casual dating to major commitments?",
        expected="First car date Nov 2020 → first formal dinner Valentine's 2021 → bought a Tesla together May 2022",
        why_matters="Requires combining 3+ memories to show relationship arc",
        # Need at least 2 of these 3 chunks in top-5
        gold_keywords=[["first car date"], ["Valentine"], ["Tesla"]],
    ),
    TestCase(
        category="Multi-Session Reasoning",
        name="Cross-cultural exchange pattern",
        question="How did Harnoor and Katie embrace each other's cultures?",
        expected="Katie: Navratri/lehenga/Garba, wore Indian clothing at Taj Mahal. Harnoor: Thanksgiving with Katie's family, Christmas together.",
        why_matters="Must combine cultural exchange moments across multiple memories",
        gold_keywords=[["Navratri", "lehenga"], ["Thanksgiving", "Katie's family"], ["Taj Mahal"]],
    ),

    # ── Temporal Reasoning ────────────────────────────────
    TestCase(
        category="Temporal Reasoning",
        name="Timeline ordering — counting repeated events",
        question="How many times did Harnoor and Katie go to Miami for New Year's?",
        expected="Twice — January 2021 and January 2022",
        why_matters="Must identify a repeated event across two different years",
        # Both Miami NYE chunks must appear
        gold_keywords=[["January 1, 2021", "Miami"], ["January 1, 2022", "Miami"]],
    ),
    TestCase(
        category="Temporal Reasoning",
        name="First trip outside Florida",
        question="When was Harnoor and Katie's first trip outside of Florida?",
        expected="July 3, 2021 — Princeton University visit in New Jersey",
        why_matters="Must distinguish Florida trips from out-of-state trips",
        gold_keywords=[["Princeton", "New Jersey"]],
    ),

    # ── Semantic Understanding ────────────────────────────
    TestCase(
        category="Semantic Understanding",
        name="Relationship milestone inference",
        question="What was the significance of Harnoor and Katie attending a wedding together?",
        expected="It signaled deeper commitment — being invited to important life events as a unit",
        why_matters="Must infer emotional/relational meaning, not just recall facts",
        gold_keywords=[["wedding", "commitment"]],
    ),
    TestCase(
        category="Semantic Understanding",
        name="Intercultural significance",
        question="Why was the Fourth of July celebration special for Harnoor?",
        expected="As an international student from India, experiencing this American tradition with Katie was culturally meaningful",
        why_matters="Must extract the cultural significance, not just 'they watched fireworks'",
        gold_keywords=[["Fourth of July", "international student"]],
    ),

    # ── Abstention / Precision ────────────────────────────
    TestCase(
        category="Abstention",
        name="Don't confuse similar events",
        question="When is Harnoor's birthday and how was it celebrated?",
        expected="December 23 — Katie surprised him with a blue gift bag at his apartment",
        why_matters="Must NOT confuse Harnoor's birthday with Katie's birthday",
        gold_keywords=[["December 23", "Harnoor's Birthday"]],
    ),
    TestCase(
        category="Abstention",
        name="Don't hallucinate engagement",
        question="Are Harnoor and Katie engaged or married?",
        expected="Unknown / not mentioned — buying a car is the biggest commitment mentioned",
        why_matters="Should NOT infer engagement/marriage from buying a car together",
        # The overview doc is the safest retrieval — no wedding/engagement signal
        gold_keywords=[["chronicles the relationship journey"]],
    ),

    # ── Negation (Vector DBs Can't "NOT") ─────────────────
    TestCase(
        category="Negation",
        name="Negation — not a birthday",
        question="What did Harnoor and Katie celebrate together that was NOT a birthday?",
        expected="Navratri, July 4th, Valentine's Day, Thanksgiving, Christmas, New Year's, wedding",
        why_matters="Vector search returns birthday chunks FIRST because 'birthday' is in the query — cosine similarity can't negate",
        # Any non-birthday celebration chunk counts as a hit
        gold_keywords=[["Navratri"], ["Fourth of July"], ["Valentine"], ["Thanksgiving"], ["Christmas Together"], ["wedding"]],
    ),
    TestCase(
        category="Negation",
        name="Negation — trips outside Florida",
        question="What trips did Harnoor and Katie take that were NOT in Florida?",
        expected="Princeton/NJ (July 2021), India (March 2022)",
        why_matters="Vector returns Miami and FSU trips (highest similarity to 'trip') even though they're IN Florida",
        gold_keywords=[["Princeton", "New Jersey"], ["Taj Mahal", "India"]],
    ),

    # ── Temporal Adjacency ────────────────────────────────
    TestCase(
        category="Temporal Adjacency",
        name="Day-after reasoning",
        question="What did Harnoor and Katie do the day after visiting Princeton?",
        expected="July 4th fireworks celebration — they watched fireworks in a large crowd",
        why_matters="Vector search has no concept of 'the day after' — returns Princeton chunk or random travel chunks",
        gold_keywords=[["Fourth of July", "fireworks"]],
    ),
    TestCase(
        category="Temporal Adjacency",
        name="Chronological ordering of milestones",
        question="Put these events in chronological order: buying a car, visiting India, attending a wedding",
        expected="Wedding (Aug 2021) → India (March 2022) → Tesla (May 2022)",
        why_matters="Vector returns chunks in SIMILARITY order, not chronological — embeddings have no clock",
        # All 3 event chunks must be in top 5
        gold_keywords=[["wedding"], ["Taj Mahal"], ["Tesla"]],
    ),

    # ── Entity Direction (WHO did WHAT to WHOM) ───────────
    TestCase(
        category="Entity Direction",
        name="Directional gift giving",
        question="What did Katie specifically do for Harnoor's birthday?",
        expected="Surprised him with a large blue 'Happy Birthday' gift bag at his apartment on December 23, 2020",
        why_matters="Vector returns both birthday chunks equally — can't distinguish Katie→Harnoor from Harnoor→Katie direction",
        gold_keywords=[["December 23", "Katie surprised"]],
    ),
    TestCase(
        category="Entity Direction",
        name="One-directional cultural adoption",
        question="Which of Harnoor's Indian traditions did Katie participate in?",
        expected="Navratri (wore lehenga, danced Garba), wore Indian clothing at Taj Mahal",
        why_matters="Vector also returns Thanksgiving/Christmas (Harnoor adopting Katie's traditions) — can't filter by direction",
        gold_keywords=[["Navratri", "lehenga"], ["Taj Mahal", "Indian"]],
    ),

    # ── Geographic Entity Filtering ───────────────────────
    TestCase(
        category="Geographic Filtering",
        name="International vs domestic",
        question="Which events happened outside of the United States?",
        expected="Only the India trip — visiting the Taj Mahal in March 2022",
        why_matters="Vector returns any travel chunk (Miami, Princeton, India all score similarly on 'events outside') — no geographic entity awareness",
        gold_keywords=[["Taj Mahal", "India"]],
    ),

    # ── Aggregation Across All Chunks ─────────────────────
    TestCase(
        category="Aggregation",
        name="Exhaustive listing",
        question="List every holiday or celebration Harnoor and Katie shared together",
        expected="Harnoor's birthday, Katie's birthday, NYE 2021, NYE 2022, Valentine's Day, Navratri, July 4th, Thanksgiving, Christmas, wedding attendance",
        why_matters="Vector top-5 retrieval returns ~5 most 'celebration-like' chunks and misses the rest — can't aggregate across all 19 memories",
        # Each distinct celebration chunk is a gold group
        gold_keywords=[
            ["Harnoor's Birthday"], ["Katie's Birthday"], ["Miami", "2021"],
            ["Miami", "2022"], ["Valentine"], ["Navratri"],
            ["Fourth of July"], ["Thanksgiving"], ["Christmas Together"], ["wedding"],
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════
#  PDF TEXT EXTRACTION — same text for both systems
# ═══════════════════════════════════════════════════════════════

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "InputData", "relationship_timeline.pdf")


def extract_pdf_chunks(pdf_path: str = PDF_PATH) -> list[str]:
    """Extract one text chunk per page from the PDF."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            # Normalize whitespace (PDF line breaks → single spaces)
            text = " ".join(text.split())
            chunks.append(text)
    doc.close()
    return chunks


# ═══════════════════════════════════════════════════════════════
#  LLM-AS-JUDGE (Gemini 2.5 Flash)
# ═══════════════════════════════════════════════════════════════

def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env for LLM-as-judge evaluation")
    return genai.Client(api_key=api_key)


def generate_answer(client, question: str, chunks: list[str]) -> str:
    """Use Gemini to generate a one-sentence answer from retrieved chunks."""
    if not chunks:
        return "(no chunks retrieved)"
    context = "\n\n---\n\n".join(chunks[:5])
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=(
            f"Based ONLY on these retrieved memories, answer the question in ONE concise sentence. "
            f"If the memories don't contain the answer, say 'Not found in memories.'\n\n"
            f"Question: {question}\n\n"
            f"Retrieved memories:\n{context}"
        ),
    )
    return resp.text.strip()


def judge_answer(client, question: str, expected: str, ai_answer: str) -> dict:
    """
    LLM-as-judge: score the AI answer against the expected answer.
    Returns {"verdict": "YES"/"NO", "score": 1-10, "reasoning": str}
    """
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=(
            f"You are a strict evaluator. Compare the AI answer against the expected answer.\n\n"
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"AI answer: {ai_answer}\n\n"
            f"Evaluate:\n"
            f"1. Does the AI answer contain the KEY FACTS from the expected answer? (YES/NO)\n"
            f"2. Score 1-10 for completeness and accuracy.\n\n"
            f"Respond in EXACTLY this JSON format, no other text:\n"
            f'{{"verdict": "YES or NO", "score": <1-10>, "reasoning": "<one sentence>"}}'
        ),
    )
    text = resp.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"verdict": "NO", "score": 0, "reasoning": f"Parse error: {text[:200]}"}


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
    raw_chunks: list[str] = None  # full text of each retrieved chunk

    def __post_init__(self):
        if self.raw_chunks is None:
            self.raw_chunks = []


# ═══════════════════════════════════════════════════════════════
#  EVAL METRICS — recall@5 and MRR
# ═══════════════════════════════════════════════════════════════

def _chunk_matches_group(chunk_text: str, keyword_group: list[str]) -> bool:
    """True if ALL keywords in the group appear in chunk_text (case-insensitive)."""
    text_lower = chunk_text.lower()
    return all(kw.lower() in text_lower for kw in keyword_group)


def compute_recall_at_k(chunks: list[str], gold_groups: list[list[str]], k: int = 5) -> float:
    """
    Recall@K: fraction of gold keyword groups matched by at least one chunk in top-K.
    Each gold group represents one piece of required information.
    A group is "recalled" if any chunk in the top-K contains ALL its keywords.
    """
    if not gold_groups:
        return 0.0
    top_k = chunks[:k]
    recalled = 0
    for group in gold_groups:
        if any(_chunk_matches_group(c, group) for c in top_k):
            recalled += 1
    return recalled / len(gold_groups)


def compute_mrr(chunks: list[str], gold_groups: list[list[str]]) -> float:
    """
    MRR (Mean Reciprocal Rank): 1/rank of the FIRST chunk that matches ANY gold group.
    If no chunk matches, MRR = 0.
    """
    if not gold_groups:
        return 0.0
    for rank, chunk in enumerate(chunks, 1):
        if any(_chunk_matches_group(chunk, group) for group in gold_groups):
            return 1.0 / rank
    return 0.0


# ═══════════════════════════════════════════════════════════════
#  CHROMADB RUNNER
# ═══════════════════════════════════════════════════════════════

def run_chroma(chunks: list[str]) -> tuple[list[Result], ChromaClient]:
    """Ingest PDF text chunks into ChromaDB, then run queries."""
    print("\n  [Vector DB] Initializing local vector database...")
    client = ChromaClient(collection_name="relationship_timeline")

    # Ingest all PDF text chunks
    print(f"  [Vector DB] Ingesting {len(chunks)} PDF text chunks...")
    client.add_memories(chunks)
    print(f"  [Vector DB] {client.count()} chunks indexed (instant — local embeddings)")

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        start = time.time()
        try:
            hits = client.search(case.question, top_k=10)
            latency = (time.time() - start) * 1000

            raw_chunks = [h["text"] for h in hits] if hits else []
            if hits:
                answer = " | ".join(h["text"][:150] for h in hits[:3])
            else:
                answer = "(no results)"

            print(f"{latency:.1f}ms ({len(hits)} chunks)")
            results.append(Result(
                "Traditional Vector DB", case.name, case.category,
                answer[:500], latency, len(hits), "", raw_chunks,
            ))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result(
                "Traditional Vector DB", case.name, case.category,
                "", latency, 0, str(e), [],
            ))

    return results, client


# ═══════════════════════════════════════════════════════════════
#  HYDRADB RUNNER
# ═══════════════════════════════════════════════════════════════

def run_hydradb(
    api_key: str,
    chunks: list[str],
    tenant_id: str = "test1",
    sub_tenant_id: str = "opj3bivvmh",
    skip_ingest: bool = False,
) -> list[Result]:
    """Run queries against HydraDB using full_recall (hybrid vector+graph+BM25)."""
    client = HydraDBClient(api_key=api_key)

    # ── Fresh tenant + chunk ingestion ──────────────────────
    print(f"\n  [HydraDB] Using tenant: {tenant_id}, sub_tenant: {sub_tenant_id}")
    client.use_tenant(tenant_id)

    if not skip_ingest:
        print(f"  [HydraDB] Ingesting {len(chunks)} PDF text chunks one-by-one...")
        for i, chunk in enumerate(chunks):
            result = client.add_memory(chunk, infer=True)
            status = result.get("results", [{}])[0].get("status", "?")
            print(f"    [{i+1}/{len(chunks)}] {status} ({len(chunk)} chars)")
            time.sleep(1)  # Small delay between adds
        print("  [HydraDB] All chunks submitted. Waiting 60s for indexing...")
        time.sleep(60)

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case.name}...", end=" ", flush=True)

        start = time.time()
        try:
            resp = client.recall_preferences(
                query=case.question,
                max_results=10,
                sub_tenant_id=sub_tenant_id,
                graph_context=True,
            )
            latency = (time.time() - start) * 1000

            resp_chunks = resp.get("chunks", [])
            raw_chunks = [c.get("chunk_content", "") for c in resp_chunks]
            if resp_chunks:
                answer = " | ".join(
                    c.get("chunk_content", "")[:150] for c in resp_chunks[:3]
                )
            else:
                answer = str(resp)[:200]

            print(f"{latency:.0f}ms ({len(resp_chunks)} chunks)")
            results.append(Result(
                "HydraDB", case.name, case.category,
                answer[:500], latency, len(resp_chunks), "", raw_chunks,
            ))
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"ERR: {e}")
            results.append(Result(
                "HydraDB", case.name, case.category,
                "", latency, 0, str(e), [],
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

    # ── Extract PDF text (same data for both systems) ─────
    pdf_chunks = extract_pdf_chunks()

    print("=" * 70)
    print("  BENCHMARK: HydraDB vs Traditional Vector DB")
    print(f"  {len(TEST_CASES)} test cases | {len(pdf_chunks)} PDF text chunks")
    print(f"  Source: relationship_timeline.pdf (text extracted, identical for both)")
    print(f"  Vector DB: local, pure cosine similarity, standard embeddings")
    print(f"  HydraDB:  cloud, hybrid vector+graph+BM25")
    print("=" * 70)

    # ── Run ChromaDB ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING TRADITIONAL VECTOR DB (Pure Vector)")
    print(f"{'─'*70}")
    chroma_results, chroma_client = run_chroma(pdf_chunks)

    # ── Run HydraDB ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RUNNING HYDRADB (Recall Preferences — hybrid vector+graph+BM25)")
    print(f"{'─'*70}")
    hydra_results = run_hydradb(
        hydra_key, pdf_chunks,
        tenant_id="test1",
        sub_tenant_id="opj3bivvmh",
        skip_ingest=True,  # Data already uploaded as memories
    )

    # ── LLM-as-Judge Evaluation ────────────────────────────
    print(f"\n{'='*70}")
    print("  LLM-AS-JUDGE EVALUATION (Gemini 3 Flash)")
    print(f"{'='*70}")

    gemini = _get_gemini_client()

    comparison = []
    h_scores, c_scores = [], []
    h_yeses, c_yeses = 0, 0

    for i, (case, chroma, hydra) in enumerate(zip(TEST_CASES, chroma_results, hydra_results)):
        print(f"\n  [{i+1}/{len(TEST_CASES)}] {case.name}")
        print(f"  │  Q: {case.question}")

        # Generate AI answers from top-5 chunks
        h_ai_answer = generate_answer(gemini, case.question, hydra.raw_chunks)
        c_ai_answer = generate_answer(gemini, case.question, chroma.raw_chunks)

        # Judge both answers
        h_judgment = judge_answer(gemini, case.question, case.expected, h_ai_answer)
        c_judgment = judge_answer(gemini, case.question, case.expected, c_ai_answer)

        h_score = h_judgment.get("score", 0)
        c_score = c_judgment.get("score", 0)
        h_verdict = h_judgment.get("verdict", "NO")
        c_verdict = c_judgment.get("verdict", "NO")

        h_scores.append(h_score)
        c_scores.append(c_score)
        if h_verdict == "YES":
            h_yeses += 1
        if c_verdict == "YES":
            c_yeses += 1

        # Determine winner from judge score
        if h_score > c_score:
            winner = "hydra"
        elif c_score > h_score:
            winner = "chroma"
        else:
            winner = "tie"

        # Also compute retrieval metrics
        h_recall5 = compute_recall_at_k(hydra.raw_chunks, case.gold_keywords, k=5)
        c_recall5 = compute_recall_at_k(chroma.raw_chunks, case.gold_keywords, k=5)
        h_recall10 = compute_recall_at_k(hydra.raw_chunks, case.gold_keywords, k=10)
        c_recall10 = compute_recall_at_k(chroma.raw_chunks, case.gold_keywords, k=10)

        print(f"  ├─ HydraDB:  {h_verdict} ({h_score}/10) — {h_ai_answer[:100]}")
        print(f"  ├─ ChromaDB: {c_verdict} ({c_score}/10) — {c_ai_answer[:100]}")
        print(f"  └─ Winner: {winner.upper()}")

        comparison.append({
            "category": case.category,
            "name": case.name,
            "question": case.question,
            "expected": case.expected,
            "hydra_ai_answer": h_ai_answer,
            "hydra_verdict": h_verdict,
            "hydra_score": h_score,
            "hydra_reasoning": h_judgment.get("reasoning", ""),
            "hydra_latency_ms": round(hydra.latency_ms),
            "hydra_chunks": hydra.chunks_returned,
            "hydra_recall_at_5": round(h_recall5, 4),
            "hydra_recall_at_10": round(h_recall10, 4),
            "chroma_ai_answer": c_ai_answer,
            "chroma_verdict": c_verdict,
            "chroma_score": c_score,
            "chroma_reasoning": c_judgment.get("reasoning", ""),
            "chroma_latency_ms": round(chroma.latency_ms, 1),
            "chroma_chunks": chroma.chunks_returned,
            "chroma_recall_at_5": round(c_recall5, 4),
            "chroma_recall_at_10": round(c_recall10, 4),
            "winner": winner,
            "why_matters": case.why_matters,
        })

        time.sleep(0.5)  # Rate limit courtesy

    # ── Aggregate metrics ─────────────────────────────────
    wins_h = sum(1 for c in comparison if c["winner"] == "hydra")
    wins_c = sum(1 for c in comparison if c["winner"] == "chroma")
    ties = sum(1 for c in comparison if c["winner"] == "tie")

    avg_h_score = sum(h_scores) / len(h_scores)
    avg_c_score = sum(c_scores) / len(c_scores)

    c_lats = [r.latency_ms for r in chroma_results]
    h_lats = [r.latency_ms for r in hydra_results]

    summary = {
        "chromadb": {
            "avg_latency_ms": round(sum(c_lats) / len(c_lats), 1),
            "p50_latency_ms": round(sorted(c_lats)[len(c_lats) // 2], 1),
            "min_latency_ms": round(min(c_lats), 1),
            "max_latency_ms": round(max(c_lats), 1),
            "total": len(chroma_results),
            "type": "Pure vector (cosine similarity)",
            "location": "Local (in-memory)",
            "avg_score": round(avg_c_score, 2),
            "yes_count": c_yeses,
            "wins": wins_c,
        },
        "hydradb": {
            "avg_latency_ms": round(sum(h_lats) / len(h_lats)),
            "p50_latency_ms": round(sorted(h_lats)[len(h_lats) // 2]),
            "min_latency_ms": round(min(h_lats)),
            "max_latency_ms": round(max(h_lats)),
            "total": len(hydra_results),
            "type": "Hybrid (vector + graph + BM25)",
            "location": "Cloud API",
            "avg_score": round(avg_h_score, 2),
            "yes_count": h_yeses,
            "wins": wins_h,
        },
        "ties": ties,
        "judge_model": "gemini-3-flash-preview",
    }

    print(f"\n{'='*70}")
    print("  LLM-AS-JUDGE RESULTS")
    print(f"{'='*70}")
    print(f"  {'':25s} {'Avg Score':>10s} {'YES':>6s} {'Wins':>8s}")
    print(f"  {'HydraDB (hybrid)':25s} {avg_h_score:>8.1f}/10 {h_yeses:>5d}/{len(comparison)} {wins_h:>6d}/{len(comparison)}")
    print(f"  {'ChromaDB (pure vector)':25s} {avg_c_score:>8.1f}/10 {c_yeses:>5d}/{len(comparison)} {wins_c:>6d}/{len(comparison)}")
    print(f"  {'Ties':25s} {'':>10s} {'':>6s} {ties:>6d}/{len(comparison)}")

    print(f"\n{'='*70}")
    print("  LATENCY")
    print(f"{'='*70}")
    print(f"  {'':25s} {'Avg':>10s} {'P50':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'ChromaDB (local)':25s} {summary['chromadb']['avg_latency_ms']:>8.1f}ms {summary['chromadb']['p50_latency_ms']:>8.1f}ms {summary['chromadb']['min_latency_ms']:>8.1f}ms {summary['chromadb']['max_latency_ms']:>8.1f}ms")
    print(f"  {'HydraDB (cloud)':25s} {summary['hydradb']['avg_latency_ms']:>8d}ms {summary['hydradb']['p50_latency_ms']:>8d}ms {summary['hydradb']['min_latency_ms']:>8d}ms {summary['hydradb']['max_latency_ms']:>8d}ms")

    print(f"\n  Judge: Gemini 3 Flash | Both systems got the same {len(pdf_chunks)} memory chunks.")

    # ── Save ──────────────────────────────────────────────
    output = {
        "summary": summary,
        "comparison": comparison,
        "data": {
            "total_chunks_ingested": len(pdf_chunks),
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
