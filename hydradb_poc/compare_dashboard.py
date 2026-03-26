"""
Streamlit Dashboard: HydraDB vs Mem0 — Side-by-Side Comparison

Run:
  streamlit run hydradb_poc/compare_dashboard.py

Modes:
  1. Live mode  — enter API keys, ingest data, run benchmark in real-time
  2. Results mode — load saved compare_results.json from a previous run
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="HydraDB vs Mem0 Benchmark", layout="wide")

st.title("HydraDB vs Mem0 — Memory Benchmark")
st.caption("Side-by-side comparison on a real relationship timeline dataset")


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", ["Live Benchmark", "View Saved Results"])

    if mode == "Live Benchmark":
        st.subheader("API Keys")
        hydradb_key = st.text_input(
            "HydraDB API Key",
            value=os.getenv("HYDRADB_API_KEY", ""),
            type="password",
        )
        mem0_key = st.text_input(
            "Mem0 API Key",
            value=os.getenv("MEM0_API_KEY", ""),
            type="password",
        )
        st.divider()
        st.markdown("""
        **Steps:**
        1. Enter API keys above
        2. Click "Ingest Data" to upload to both systems
        3. Click "Run Benchmark" to compare
        """)
    else:
        results_file = os.path.join(os.path.dirname(__file__), "compare_results.json")
        st.info(f"Loading from: compare_results.json")


# ── Category colors ──────────────────────────────────────────
CAT_COLORS = {
    "Information Extraction": "#4CAF50",
    "Multi-Fact Reasoning": "#2196F3",
    "Temporal Reasoning": "#FF9800",
    "Knowledge Updates": "#9C27B0",
    "Abstention": "#F44336",
}


# ═══════════════════════════════════════════════════════════════
#  LIVE BENCHMARK MODE
# ═══════════════════════════════════════════════════════════════

if mode == "Live Benchmark":
    from hydradb_poc.client import HydraDBClient
    from hydradb_poc.mem0_client import Mem0Client
    from hydradb_poc.ingest import ingest_hydradb, ingest_mem0, TIMELINE_CHUNKS, PDF_PATH
    from hydradb_poc.compare_benchmark import (
        QUESTIONS, query_hydradb, query_mem0, BenchmarkResult, save_results,
    )

    # ── Step 1: Ingest Data ──────────────────────────────────
    st.markdown("---")
    st.header("Step 1: Ingest Data")
    st.markdown("""
    Upload the **relationship timeline PDF** to HydraDB and ingest text chunks to Mem0.
    Both systems receive the same content — 18 events spanning Oct 2020 to May 2022.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("HydraDB")
        st.markdown("- Uploads the full **PDF** via `/ingestion/upload_knowledge`")
        st.markdown("- HydraDB extracts text, builds vector index + knowledge graph")
        st.markdown(f"- File: `{os.path.basename(PDF_PATH)}`")

    with col2:
        st.subheader("Mem0")
        st.markdown(f"- Ingests **{len(TIMELINE_CHUNKS)} text chunks** via `/v1/memories/`")
        st.markdown("- Mem0 extracts facts and builds vector embeddings")
        st.markdown("- Same content as the PDF, pre-chunked")

    if st.button("Ingest Data", type="primary", disabled=not (hydradb_key or mem0_key)):
        with st.status("Ingesting data into both systems...", expanded=True) as status:
            # HydraDB
            if hydradb_key:
                st.write("**HydraDB:** Uploading PDF...")
                hydra = HydraDBClient(api_key=hydradb_key)
                try:
                    hydra.create_tenant("relationship_benchmark")
                except Exception:
                    hydra.use_tenant("relationship_benchmark")
                try:
                    result = ingest_hydradb(hydra)
                    st.write(f"**HydraDB:** Upload complete! Response: `{json.dumps(result)[:200]}`")
                    st.session_state["hydra_ingested"] = True
                except Exception as e:
                    st.error(f"HydraDB ingestion failed: {e}")
            else:
                st.warning("No HydraDB API key — skipping")

            # Mem0
            if mem0_key:
                st.write(f"**Mem0:** Ingesting {len(TIMELINE_CHUNKS)} chunks...")
                mem0 = Mem0Client(api_key=mem0_key, user_id="relationship_benchmark")
                try:
                    results = ingest_mem0(mem0)
                    errors = sum(1 for r in results if "error" in r)
                    st.write(f"**Mem0:** Done! {len(results) - errors}/{len(results)} chunks ingested.")
                    st.session_state["mem0_ingested"] = True
                except Exception as e:
                    st.error(f"Mem0 ingestion failed: {e}")
            else:
                st.warning("No Mem0 API key — skipping")

            status.update(label="Ingestion complete!", state="complete")

    # ── Step 2: Run Benchmark ────────────────────────────────
    st.markdown("---")
    st.header("Step 2: Run Benchmark")
    st.markdown(f"**{len(QUESTIONS)} questions** across **{len(set(q.category for q in QUESTIONS))} categories**")

    # Show questions preview
    with st.expander("Preview all benchmark questions"):
        for i, q in enumerate(QUESTIONS):
            color = CAT_COLORS.get(q.category, "#666")
            st.markdown(f"**{i+1}.** <span style='color:{color}'>[{q.category}]</span> {q.question}",
                        unsafe_allow_html=True)
            st.caption(f"Expected: {q.expected_answer[:150]}...")

    if st.button("Run Benchmark", type="primary", disabled=not (hydradb_key or mem0_key)):
        hydra = None
        mem0 = None

        if hydradb_key:
            hydra = HydraDBClient(api_key=hydradb_key)
            hydra.use_tenant("relationship_benchmark")
        if mem0_key:
            mem0 = Mem0Client(api_key=mem0_key, user_id="relationship_benchmark")

        results = []
        progress = st.progress(0)

        for i, q in enumerate(QUESTIONS):
            with st.status(f"[{i+1}/{len(QUESTIONS)}] {q.category}: {q.question[:60]}...") as status:
                result = BenchmarkResult(question=q)

                if hydra:
                    st.write("Querying HydraDB...")
                    result.hydra_answer, result.hydra_latency_ms, result.hydra_sources_count = \
                        query_hydradb(hydra, q.question)

                if mem0:
                    st.write("Querying Mem0...")
                    result.mem0_answer, result.mem0_latency_ms, result.mem0_sources_count = \
                        query_mem0(mem0, q.question)

                results.append(result)
                status.update(label=f"Done: {q.question[:60]}...", state="complete")

            progress.progress((i + 1) / len(QUESTIONS))

        # Save results
        output_path = os.path.join(os.path.dirname(__file__), "compare_results.json")
        saved = save_results(results, output_path)
        st.session_state["benchmark_results"] = results
        st.session_state["benchmark_data"] = saved
        st.success("Benchmark complete! Results saved.")

    # ── Show Results ─────────────────────────────────────────
    if "benchmark_results" in st.session_state:
        results = st.session_state["benchmark_results"]
        _show_results(results)


# ═══════════════════════════════════════════════════════════════
#  VIEW SAVED RESULTS MODE
# ═══════════════════════════════════════════════════════════════

def _show_results(data):
    """Display benchmark results (works for both live and saved)."""
    # Handle both live BenchmarkResult objects and saved JSON dicts
    if isinstance(data, list) and len(data) > 0:
        if hasattr(data[0], "question"):
            # Live BenchmarkResult objects — convert to dicts
            items = []
            for r in data:
                items.append({
                    "category": r.question.category,
                    "question": r.question.question,
                    "expected": r.question.expected_answer,
                    "why_hard": r.question.why_hard,
                    "hydra_answer": r.hydra_answer,
                    "hydra_latency_ms": r.hydra_latency_ms,
                    "hydra_sources": r.hydra_sources_count,
                    "mem0_answer": r.mem0_answer,
                    "mem0_latency_ms": r.mem0_latency_ms,
                    "mem0_sources": r.mem0_sources_count,
                })
        else:
            items = data
    else:
        items = data.get("results", []) if isinstance(data, dict) else []

    if not items:
        st.warning("No results to display.")
        return

    st.markdown("---")
    st.header("Results")

    # ── Summary metrics ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(items))
    with col2:
        avg_h = sum(r["hydra_latency_ms"] for r in items) / len(items)
        st.metric("HydraDB Avg Latency", f"{avg_h:.0f}ms")
    with col3:
        avg_m = sum(r["mem0_latency_ms"] for r in items) / len(items)
        st.metric("Mem0 Avg Latency", f"{avg_m:.0f}ms")

    # ── Category breakdown ───────────────────────────────────
    st.subheader("Latency by Category")
    categories = {}
    for r in items:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"hydra": [], "mem0": []}
        categories[cat]["hydra"].append(r["hydra_latency_ms"])
        categories[cat]["mem0"].append(r["mem0_latency_ms"])

    cols = st.columns(len(categories))
    for col, (cat, data) in zip(cols, categories.items()):
        with col:
            h_avg = sum(data["hydra"]) / len(data["hydra"])
            m_avg = sum(data["mem0"]) / len(data["mem0"])
            st.markdown(f"**{cat}**")
            st.markdown(f"HydraDB: `{h_avg:.0f}ms`")
            st.markdown(f"Mem0: `{m_avg:.0f}ms`")

    # ── Detailed results ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Question-by-Question Comparison")

    for i, r in enumerate(items):
        color = CAT_COLORS.get(r["category"], "#666")
        with st.expander(
            f"Q{i+1}: [{r['category']}] {r['question'][:80]}...",
            expanded=(i < 3),
        ):
            st.markdown(f"**Category:** <span style='color:{color}'>{r['category']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Question:** {r['question']}")
            st.markdown(f"**Expected Answer:** {r['expected']}")
            st.markdown(f"**Why this is hard:** _{r['why_hard']}_")

            st.markdown("---")
            col_h, col_m = st.columns(2)

            with col_h:
                st.markdown("#### HydraDB")
                st.markdown(f"**Latency:** `{r['hydra_latency_ms']}ms` | **Sources:** `{r['hydra_sources']}`")
                if r["hydra_answer"]:
                    st.text_area(
                        "HydraDB Answer",
                        r["hydra_answer"],
                        height=200,
                        disabled=True,
                        key=f"hydra_{i}",
                    )
                else:
                    st.info("No HydraDB response (API key not provided)")

            with col_m:
                st.markdown("#### Mem0")
                st.markdown(f"**Latency:** `{r['mem0_latency_ms']}ms` | **Sources:** `{r['mem0_sources']}`")
                if r["mem0_answer"]:
                    st.text_area(
                        "Mem0 Answer",
                        r["mem0_answer"],
                        height=200,
                        disabled=True,
                        key=f"mem0_{i}",
                    )
                else:
                    st.info("No Mem0 response (API key not provided)")

    # ── Architecture comparison ──────────────────────────────
    st.markdown("---")
    st.subheader("Architecture Comparison")
    st.markdown("""
    | Feature | **HydraDB** | **Mem0** |
    |---|---|---|
    | **Input** | PDF upload (handles parsing) | Text memories (pre-chunked) |
    | **Retrieval** | Hybrid: vector + knowledge graph + BM25 | Vector similarity search |
    | **Output** | AI-generated answer + source chunks | Ranked memory results |
    | **Graph** | Built-in knowledge graph with entity linking | No graph (flat memory store) |
    | **Temporal** | Versioned graph with temporal edges | No temporal tracking |
    | **Best for** | Complex Q&A, multi-hop reasoning, documents | Simple fact recall, user preferences |
    """)


if mode == "View Saved Results":
    results_file = os.path.join(os.path.dirname(__file__), "compare_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        st.success(f"Loaded {len(data.get('results', []))} results from {data.get('timestamp', 'unknown')}")
        _show_results(data)
    else:
        st.warning("No saved results found. Run the benchmark first:")
        st.code("python -m hydradb_poc.compare_benchmark --live", language="bash")
        st.markdown("Or use **Live Benchmark** mode in the sidebar.")

        # Show the questions anyway
        from hydradb_poc.compare_benchmark import QUESTIONS
        st.markdown("---")
        st.header("Benchmark Questions Preview")
        for i, q in enumerate(QUESTIONS):
            color = CAT_COLORS.get(q.category, "#666")
            st.markdown(f"**{i+1}.** <span style='color:{color}'>[{q.category}]</span> {q.question}",
                        unsafe_allow_html=True)
            st.caption(f"Expected: {q.expected_answer[:150]}...")
