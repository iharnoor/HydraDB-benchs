"""
╔══════════════════════════════════════════════════════════════════╗
║  Benchmark Dashboard — Why HydraDB > Plain RAG                   ║
║                                                                  ║
║  Visual proof that graph-based memory beats flat vector stores   ║
║                                                                  ║
║  Run: streamlit run rag/hydradb_poc/dashboard.py                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="HydraDB Benchmarks", layout="wide")

# ── Load or generate benchmark data ─────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "benchmark_data.json")

if not os.path.exists(DATA_PATH):
    from hydradb_poc.benchmark import run_offline_benchmark
    data = run_offline_benchmark()
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
else:
    with open(DATA_PATH) as f:
        data = json.load(f)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════

st.title("HydraDB vs The Competition")
st.caption("LongMemEval-s Benchmark | 500 questions | Avg 115k tokens per conversation stack")

st.markdown("""
> **Why this matters:** Plain RAG (chunk → embed → cosine search) breaks down on real-world
> memory tasks. HydraDB's temporal graph + sliding window + multi-hop recall achieves
> **90.79% accuracy** — a 30-point lead over full-context baselines.
""")

# ══════════════════════════════════════════════════════════════
#  OVERALL SCORES — BIG NUMBER CARDS
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Overall Accuracy")

cols = st.columns(5)
overall = data["overall"]
sorted_systems = sorted(overall.items(), key=lambda x: -x[1])
colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

for i, (system, score) in enumerate(sorted_systems):
    with cols[i]:
        delta = f"+{score - 60.2:.1f}% vs baseline" if system == "HydraDB" else None
        st.metric(system, f"{score:.1f}%", delta=delta)


# ══════════════════════════════════════════════════════════════
#  CATEGORY BREAKDOWN — GROUPED BAR CHART
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Performance by Category")
st.markdown("Each bar shows accuracy on a specific memory capability. "
            "HydraDB leads across **every** category.")

# Build dataframe for chart
rows = []
for system, scores in data["systems"].items():
    for cat, score in zip(data["categories_clean"], scores):
        rows.append({"System": system, "Category": cat, "Accuracy (%)": score})

df = pd.DataFrame(rows)

# Streamlit native bar chart (pivot for grouped display)
pivot = df.pivot(index="Category", columns="System", values="Accuracy (%)")
# Reorder columns by overall score
pivot = pivot[["HydraDB", "Supermemory", "Zep", "Full-Context (GPT-4o)", "Mem0-OSS"]]
st.bar_chart(pivot, height=450)


# ══════════════════════════════════════════════════════════════
#  WHERE HYDRADB WINS THE MOST
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Where HydraDB Wins Big")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Temporal Reasoning")
    st.metric("HydraDB", "90.97%")
    st.metric("Full-Context (GPT-4o)", "45.1%", delta="-45.9%", delta_color="inverse")
    st.markdown("""
    **Why?** HydraDB's Git-style versioned graph
    tracks when facts became true/false.
    Plain RAG has no concept of time.

    ```
    User lives in NYC (2022)
         ──SUPERSEDED_BY──►
    User lives in London (2024)
    ```
    """)

with col2:
    st.subheader("Preference Understanding")
    st.metric("HydraDB", "96.67%")
    st.metric("Full-Context (GPT-4o)", "20.0%", delta="-76.7%", delta_color="inverse")
    st.markdown("""
    **Why?** Sliding Window Inference captures
    implicit preferences, not just keywords.

    ```
    "nothing beats a graph database"
         ──INFERRED──►
    preference: graph_databases (strong)
    ```
    """)

with col3:
    st.subheader("Knowledge Updates")
    st.metric("HydraDB", "97.43%")
    st.metric("Full-Context (GPT-4o)", "78.2%", delta="-19.2%", delta_color="inverse")
    st.markdown("""
    **Why?** Append-only ledger preserves ALL
    states. No destructive overwrites.

    ```
    diet: omnivore (t1)
         ──UPDATED_TO──►
    diet: vegetarian (t2, reason: health)
    ```
    """)


# ══════════════════════════════════════════════════════════════
#  ARCHITECTURE COMPARISON
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Architecture: Why the Difference?")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Plain RAG")
    st.code("""
┌──────────────────────────────┐
│  Document                    │
│  "I moved from NYC to London"│
└──────────┬───────────────────┘
           │ chunk
           ▼
┌──────────────────────────────┐
│  Vector Store (flat)         │
│  [0.12, -0.45, 0.78, ...]   │
│  No structure. No time.      │
│  No relationships.           │
└──────────┬───────────────────┘
           │ cosine similarity
           ▼
┌──────────────────────────────┐
│  Retrieved: "I live in NYC"  │
│  (higher similarity score!)  │
│  ❌ WRONG — that's outdated  │
└──────────────────────────────┘
    """, language="text")

    st.error("Flat vector stores can't distinguish current from outdated facts")

with col_right:
    st.subheader("HydraDB")
    st.code("""
┌──────────────────────────────┐
│  Conversation Stream          │
│  "I moved from NYC to London"│
└──────────┬───────────────────┘
           │ Sliding Window Inference
           │ + Entity Resolution
           ▼
┌──────────────────────────────┐
│  Temporal Knowledge Graph     │
│                              │
│  [User]──LIVES_IN──►[NYC]    │
│    t=2022, context: "work"   │
│  [User]──LIVES_IN──►[London] │
│    t=2024, context: "Meta"   │
│                              │
│  + Vector embeddings         │
│  + BM25 sparse index         │
└──────────┬───────────────────┘
           │ 5-stage recall pipeline
           ▼
┌──────────────────────────────┐
│  Retrieved: "Lives in London │
│  since 2024 (moved for Meta)"│
│  ✅ CORRECT — latest state   │
└──────────────────────────────┘
    """, language="text")

    st.success("Temporal graph + hybrid search = correct answers")


# ══════════════════════════════════════════════════════════════
#  CROSS-MODEL STABILITY
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Model-Agnostic Performance")
st.markdown("HydraDB's accuracy holds across different LLMs — "
            "the memory architecture does the heavy lifting, not the model.")

cross_df = pd.DataFrame([
    {"Model": model, "Accuracy (%)": score}
    for model, score in data["cross_model"].items()
])
st.bar_chart(cross_df.set_index("Model"), height=300)

st.info("Even GPT-5 Mini (a smaller model) achieves 85.8% with HydraDB — "
        "higher than GPT-4o with full context (60.2%).")


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE TEST CASES
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Benchmark Test Cases")
st.markdown("Explore the specific scenarios that break plain RAG but work with HydraDB:")

for tc in data["test_cases"]:
    with st.expander(f"**{tc['category']}** — {tc['name']}"):
        st.markdown(f"**Question:** {tc['question']}")
        st.markdown(f"**Expected Answer:** {tc['expected']}")
        st.warning(f"**Why plain RAG fails:** {tc['why_hard']}")


# ══════════════════════════════════════════════════════════════
#  RECALL PIPELINE COMPARISON
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Retrieval Pipeline: 1 Step vs 5 Stages")

st.markdown("""
| Stage | Plain RAG | HydraDB |
|-------|-----------|---------|
| **Query Processing** | Use query as-is | Adaptive Query Expansion (N reformulations) |
| **Search** | Single cosine similarity | Weighted Hybrid: dense + inferred + BM25 |
| **Graph** | None | Entity-based graph traversal + chunk expansion |
| **Reranking** | None or single-pass | Triple-tier: vector + graph + entity fusion |
| **Memory** | Stateless | User memory + Hive memory + temporal context |
| **Time Awareness** | None | Git-style versioned edges with valid-time metadata |
""")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
### Summary

HydraDB achieves **90.79%** on LongMemEval-s because it treats memory as a
**structured, temporal, relational substrate** — not a flat bag of vectors.

The three architectural pillars that make the difference:

1. **Git-Style Versioned Graph** — tracks how facts evolve over time
2. **Sliding Window Inference** — no orphaned chunks, full entity resolution
3. **5-Stage Recall Pipeline** — vector + graph + sparse + reranking + expansion

| Metric | HydraDB | Best Alternative | Improvement |
|--------|---------|-----------------|-------------|
| Overall Accuracy | 90.79% | 85.20% (Supermemory) | +5.6% |
| Temporal Reasoning | 90.97% | 81.95% (Supermemory) | +9.0% |
| Preference Understanding | 96.67% | 70.00% (Supermemory) | +26.7% |
| Knowledge Updates | 97.43% | 89.74% (Supermemory) | +7.7% |

[Try HydraDB](https://workbench.hydradb.com/) |
[Read the Paper](https://research.hydradb.com/hydradb.pdf) |
[Documentation](https://docs.hydradb.com/)
""")
