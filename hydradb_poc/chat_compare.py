"""
Side-by-side chat comparison: HydraDB vs Mem0.

Both systems recall memories from the same dataset (Harnoor & Katie timeline).
Type a question and see how each system responds in real time.
Claude generates a one-line answer from the retrieved chunks.

Run: streamlit run hydradb_poc/chat_compare.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from mem0 import MemoryClient as Mem0Client
from google import genai

from hydradb_poc.client import HydraDBClient

load_dotenv()


def init_clients():
    """Initialize all clients once."""
    if "hydra" not in st.session_state:
        hydra_key = os.getenv("HYDRADB_API_KEY", "")
        if hydra_key:
            client = HydraDBClient(api_key=hydra_key)
            client.use_tenant("live_benchmark")
            st.session_state.hydra = client
        else:
            st.session_state.hydra = None

    if "mem0" not in st.session_state:
        mem0_key = os.getenv("MEM0_API_KEY", "")
        if mem0_key:
            st.session_state.mem0 = Mem0Client(api_key=mem0_key)
        else:
            st.session_state.mem0 = None

    if "gemini" not in st.session_state:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            st.session_state.gemini = genai.Client(api_key=api_key)
        else:
            st.session_state.gemini = None

    if "history" not in st.session_state:
        st.session_state.history = []


def summarize_with_gemini(question: str, context: str) -> str:
    """Use Gemini to generate a one-line answer from retrieved memories."""
    client = st.session_state.gemini
    if not client or not context or context.startswith("*No memories") or context.startswith("Error"):
        return context

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"Based ONLY on these retrieved memories, answer the question in ONE concise sentence. "
                f"If the memories don't contain the answer, say 'Not found in memories.'\n\n"
                f"Question: {question}\n\n"
                f"Retrieved memories:\n{context}"
            ),
        )
        return resp.text
    except Exception as e:
        return f"(Gemini error: {e}) {context[:200]}"


def query_hydradb(question: str) -> tuple[str, str, float, list[dict]]:
    """Query HydraDB. Returns (answer, raw_context, latency_ms, chunks)."""
    client = st.session_state.hydra
    if not client:
        return "HydraDB API key not configured", "", 0, []
    start = time.time()
    try:
        result = client.recall_preferences(question, max_results=5)
        latency = (time.time() - start) * 1000
        chunks = result.get("chunks", [])
        if chunks:
            raw_context = "\n".join(
                c.get("chunk_content", "") for c in chunks
            )
            answer = summarize_with_gemini(question, raw_context)
        else:
            raw_context = ""
            answer = "*No memories found.*"
        return answer, raw_context, latency, chunks
    except Exception as e:
        return f"Error: {e}", "", (time.time() - start) * 1000, []


def query_mem0(question: str) -> tuple[str, str, float, list[dict]]:
    """Query Mem0. Returns (answer, raw_context, latency_ms, memories)."""
    client = st.session_state.mem0
    if not client:
        return "Mem0 API key not configured", "", 0, []
    start = time.time()
    try:
        raw = client.search(
            question,
            filters={"user_id": "relationship_benchmark"},
            limit=5,
        )
        latency = (time.time() - start) * 1000

        if isinstance(raw, dict):
            items = raw.get("results", raw.get("memories", []))
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        if items:
            memories = []
            for item in items:
                if isinstance(item, dict):
                    memories.append(item.get("memory", item.get("text", str(item))))
                else:
                    memories.append(str(item))
            raw_context = "\n".join(memories)
            answer = summarize_with_gemini(question, raw_context)
        else:
            raw_context = ""
            answer = "*No memories found.*"
        return answer, raw_context, latency, items
    except Exception as e:
        return f"Error: {e}", "", (time.time() - start) * 1000, []


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HydraDB vs Mem0 — Chat Compare",
    page_icon="\u2705",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background-color: #0f0f13; }

    .main-header {
        text-align: center;
        padding: 24px 0 16px;
    }
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #e4e4ed;
        letter-spacing: -0.02em;
    }
    .main-header h1 .hydra { color: #818cf8; }
    .main-header h1 .mem0 { color: #fbbf24; }
    .main-header h1 .vs { color: #8888a0; font-weight: 400; }
    .main-header p { color: #8888a0; font-size: 0.9rem; margin-top: 4px; }

    .system-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 6px 16px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .system-label.hydra {
        color: #818cf8;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
    }
    .system-label.mem0 {
        color: #fbbf24;
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.25);
    }

    .latency-tag {
        font-family: 'Inter', monospace;
        font-size: 0.75rem;
        color: #8888a0;
        margin-left: 8px;
    }

    .answer-bubble {
        background: #18181f;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
        color: #e4e4ed;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .raw-context {
        background: #13131a;
        border: 1px solid #222233;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        color: #8888a0;
        font-size: 0.75rem;
        line-height: 1.5;
        max-height: 120px;
        overflow-y: auto;
    }

    .chat-bubble.question {
        background: #1e1e28;
        border: 1px solid #3a3a4a;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        color: #e4e4ed;
        font-size: 0.88rem;
    }
    .chat-bubble.question .q-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8888a0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    div[data-testid="stChatInput"] textarea {
        background: #18181f !important;
        border: 1px solid #2a2a3a !important;
        color: #e4e4ed !important;
    }

    .stExpander {
        background: #13131a;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

init_clients()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1><span class="hydra">HydraDB</span> <span class="vs">vs</span> <span class="mem0">Mem0</span></h1>
    <p>Side-by-side memory recall &mdash; Harnoor &amp; Katie's relationship timeline</p>
</div>
""", unsafe_allow_html=True)

# ── Sample Questions ─────────────────────────────────────────
SAMPLES = [
    "When did Harnoor and Katie first meet?",
    "What car did they buy together?",
    "What did Katie wear for Navratri?",
    "How many times did they go to Miami?",
    "When was their first trip outside Florida?",
    "What is the significance of the Taj Mahal trip?",
    "When is Harnoor's birthday?",
    "Are they engaged or married?",
    "How did they celebrate Valentine's Day?",
    "What holidays did they celebrate together?",
]

with st.expander("Sample questions", expanded=False):
    cols = st.columns(5)
    for i, q in enumerate(SAMPLES):
        if cols[i % 5].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = q

# ── Chat Input ───────────────────────────────────────────────
question = st.chat_input("Ask something about Harnoor & Katie's relationship...")

# Handle sample button clicks
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    hydra_answer, hydra_ctx, hydra_ms, hydra_raw = query_hydradb(question)
    mem0_answer, mem0_ctx, mem0_ms, mem0_raw = query_mem0(question)

    st.session_state.history.append({
        "question": question,
        "hydra": {"answer": hydra_answer, "context": hydra_ctx, "latency": hydra_ms},
        "mem0": {"answer": mem0_answer, "context": mem0_ctx, "latency": mem0_ms},
    })

# ── Chat History ─────────────────────────────────────────────
for entry in reversed(st.session_state.history):
    # Question
    st.markdown(f"""
    <div class="chat-bubble question">
        <div class="q-label">You asked</div>
        {entry['question']}
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side answers
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div>
            <span class="system-label hydra">HydraDB</span>
            <span class="latency-tag">{entry['hydra']['latency']:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<div class="answer-bubble">{entry['hydra']['answer']}</div>""", unsafe_allow_html=True)
        if entry['hydra']['context']:
            with st.expander("Retrieved chunks", expanded=False):
                st.text(entry['hydra']['context'][:1000])

    with col2:
        st.markdown(f"""
        <div>
            <span class="system-label mem0">Mem0</span>
            <span class="latency-tag">{entry['mem0']['latency']:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<div class="answer-bubble">{entry['mem0']['answer']}</div>""", unsafe_allow_html=True)
        if entry['mem0']['context']:
            with st.expander("Retrieved memories", expanded=False):
                st.text(entry['mem0']['context'][:1000])

    st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Connection Status")

    if st.session_state.hydra:
        st.success("HydraDB connected")
        st.caption("Tenant: `live_benchmark`")
    else:
        st.error("HydraDB: No API key")

    if st.session_state.mem0:
        st.success("Mem0 connected")
        st.caption("User: `relationship_benchmark`")
    else:
        st.error("Mem0: No API key")

    if st.session_state.gemini:
        st.success("Gemini connected")
        st.caption("Generates one-line answers")
    else:
        st.warning("Gemini: No GEMINI_API_KEY (raw chunks shown)")

    st.divider()

    st.markdown("### How it works")
    st.caption(
        "1. Your question is sent to both HydraDB and Mem0\n"
        "2. Each retrieves relevant memories\n"
        "3. Gemini summarizes each into a one-line answer\n"
        "4. Raw retrieved chunks available in expandable section"
    )

    st.divider()

    st.markdown("### Endpoints")
    st.code("HydraDB: /recall/recall_preferences\nMem0: /v1/memories/search", language=None)

    if st.session_state.history:
        st.divider()
        st.markdown("### Session Stats")
        h_lats = [e["hydra"]["latency"] for e in st.session_state.history]
        m_lats = [e["mem0"]["latency"] for e in st.session_state.history]
        st.metric("Queries", len(st.session_state.history))
        c1, c2 = st.columns(2)
        c1.metric("HydraDB avg", f"{sum(h_lats)/len(h_lats):.0f}ms")
        c2.metric("Mem0 avg", f"{sum(m_lats)/len(m_lats):.0f}ms")

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
