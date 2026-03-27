"""
Side-by-side chat comparison: HydraDB vs Traditional Vector DB.

Both systems recall memories from the same dataset (Harnoor & Katie timeline).
Type a question and see how each system responds in real time.
Gemini generates a one-line answer from the retrieved chunks.

Run: streamlit run hydradb_poc/chat_compare_chroma.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai

from hydradb_poc.client import HydraDBClient
from hydradb_poc.chroma_client import ChromaClient
from hydradb_poc.ingest import TIMELINE_CHUNKS

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

    if "chroma" not in st.session_state:
        chroma = ChromaClient(collection_name="relationship_timeline")
        chroma.add_memories(TIMELINE_CHUNKS)
        st.session_state.chroma = chroma

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
            # AI-generated answer on top
            ai_answer = summarize_with_gemini(question, raw_context)
            ai_html = (
                f'<div style="margin-bottom:16px;padding:14px;background:rgba(129,140,248,0.1);border-radius:10px;border:1px solid rgba(129,140,248,0.25);">'
                f'<div style="font-size:0.7rem;font-weight:700;color:#818cf8;margin-bottom:6px;letter-spacing:0.05em;">AI ANSWER</div>'
                f'<div style="font-size:0.92rem;color:#e4e4ed;line-height:1.6;">{ai_answer}</div>'
                f'</div>'
            )
            # Format each chunk separately with score
            parts = [ai_html]
            parts.append('<div style="font-size:0.7rem;font-weight:600;color:#8888a0;margin-bottom:8px;letter-spacing:0.05em;">RETRIEVED CHUNKS</div>')
            for i, c in enumerate(chunks, 1):
                score = c.get("relevancy_score", 0)
                content = c.get("chunk_content", "")[:300]
                parts.append(
                    f'<div style="margin-bottom:12px;padding:10px;background:rgba(129,140,248,0.06);border-radius:8px;border-left:3px solid #818cf8;">'
                    f'<div style="font-size:0.72rem;font-weight:700;color:#818cf8;margin-bottom:4px;">Chunk {i} &middot; score: {score:.3f}</div>'
                    f'<div style="font-size:0.82rem;color:#e4e4ed;line-height:1.5;">{content}</div>'
                    f'</div>'
                )
            answer = "".join(parts)
        else:
            raw_context = ""
            answer = "*No memories found.*"
        return answer, raw_context, latency, chunks
    except Exception as e:
        return f"Error: {e}", "", (time.time() - start) * 1000, []


def query_chroma(question: str) -> tuple[str, str, float, list[dict]]:
    """Query ChromaDB. Returns (answer, raw_context, latency_ms, results)."""
    client = st.session_state.chroma
    if not client:
        return "Vector DB not initialized", "", 0, []
    start = time.time()
    try:
        hits = client.search(question, top_k=5)
        latency = (time.time() - start) * 1000

        if hits:
            raw_context = "\n".join(h["text"] for h in hits)
            # AI-generated answer on top
            ai_answer = summarize_with_gemini(question, raw_context)
            ai_html = (
                f'<div style="margin-bottom:16px;padding:14px;background:rgba(249,115,22,0.1);border-radius:10px;border:1px solid rgba(249,115,22,0.25);">'
                f'<div style="font-size:0.7rem;font-weight:700;color:#f97316;margin-bottom:6px;letter-spacing:0.05em;">AI ANSWER</div>'
                f'<div style="font-size:0.92rem;color:#e4e4ed;line-height:1.6;">{ai_answer}</div>'
                f'</div>'
            )
            # Format each chunk separately with score
            parts = [ai_html]
            parts.append('<div style="font-size:0.7rem;font-weight:600;color:#8888a0;margin-bottom:8px;letter-spacing:0.05em;">RETRIEVED CHUNKS</div>')
            for i, h in enumerate(hits, 1):
                score = h.get("score", 0)
                content = h["text"][:300]
                parts.append(
                    f'<div style="margin-bottom:12px;padding:10px;background:rgba(249,115,22,0.06);border-radius:8px;border-left:3px solid #f97316;">'
                    f'<div style="font-size:0.72rem;font-weight:700;color:#f97316;margin-bottom:4px;">Chunk {i} &middot; score: {score:.3f}</div>'
                    f'<div style="font-size:0.82rem;color:#e4e4ed;line-height:1.5;">{content}</div>'
                    f'</div>'
                )
            answer = "".join(parts)
        else:
            raw_context = ""
            answer = "*No memories found.*"
        return answer, raw_context, latency, hits
    except Exception as e:
        return f"Error: {e}", "", (time.time() - start) * 1000, []


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HydraDB vs Traditional Vector DB — Chat Compare",
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
    .main-header h1 .vectordb { color: #f97316; }
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
    .system-label.vectordb {
        color: #f97316;
        background: rgba(249,115,22,0.12);
        border: 1px solid rgba(249,115,22,0.25);
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
    <h1><span class="hydra">HydraDB</span> <span class="vs">vs</span> <span class="vectordb">Traditional Vector DB</span></h1>
    <p>Hybrid memory (vector + graph + BM25) vs traditional vector DB &mdash; Harnoor &amp; Katie's relationship timeline</p>
</div>
""", unsafe_allow_html=True)

# ── Sample Questions ─────────────────────────────────────────
SAMPLES = [
    "When did Harnoor and Katie first meet?",
    "What car did they buy together?",
    "What did Katie wear for Navratri?",
    "How many times did they go to Miami?",
    "When was their first trip outside Florida?",
    "What did they do the day after visiting Princeton?",
    "What did they celebrate that was NOT a birthday?",
    "What trips were NOT in Florida?",
    "Which Indian traditions did Katie participate in?",
    "Are they engaged or married?",
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
    chroma_answer, chroma_ctx, chroma_ms, chroma_raw = query_chroma(question)

    st.session_state.history.append({
        "question": question,
        "hydra": {"answer": hydra_answer, "context": hydra_ctx, "latency": hydra_ms},
        "chroma": {"answer": chroma_answer, "context": chroma_ctx, "latency": chroma_ms},
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
            <span class="system-label vectordb">Traditional Vector DB</span>
            <span class="latency-tag">{entry['chroma']['latency']:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<div class="answer-bubble">{entry['chroma']['answer']}</div>""", unsafe_allow_html=True)
        if entry['chroma']['context']:
            with st.expander("Retrieved chunks", expanded=False):
                st.text(entry['chroma']['context'][:1000])

    st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Connection Status")

    if st.session_state.hydra:
        st.success("HydraDB connected")
        st.caption("Tenant: `live_benchmark`")
    else:
        st.error("HydraDB: No API key")

    if st.session_state.chroma:
        st.success("Vector DB loaded")
        st.caption(f"{st.session_state.chroma.count()} chunks indexed (local)")
    else:
        st.error("Vector DB: Not initialized")

    if st.session_state.gemini:
        st.success("Gemini connected")
        st.caption("Generates one-line answers")
    else:
        st.warning("Gemini: No GEMINI_API_KEY (raw chunks shown)")

    st.divider()

    st.markdown("### How it works")
    st.caption(
        "1. Your question is sent to both HydraDB and a Traditional Vector DB\n"
        "2. HydraDB: hybrid vector + graph + BM25 (cloud API)\n"
        "3. Vector DB: pure cosine similarity (local embeddings)\n"
        "4. Gemini summarizes each into a one-line answer\n"
        "5. Raw retrieved chunks available in expandable section"
    )

    st.divider()

    st.markdown("### System Comparison")
    st.markdown("""
    | | **HydraDB** | **Traditional Vector DB** |
    |---|---|---|
    | **Type** | Hybrid | Pure vector |
    | **Search** | Vector+Graph+BM25 | Cosine similarity |
    | **Location** | Cloud API | Local (in-memory) |
    | **Embeddings** | Proprietary | Standard embeddings |
    """)

    if st.session_state.history:
        st.divider()
        st.markdown("### Session Stats")
        h_lats = [e["hydra"]["latency"] for e in st.session_state.history]
        c_lats = [e["chroma"]["latency"] for e in st.session_state.history]
        st.metric("Queries", len(st.session_state.history))
        c1, c2 = st.columns(2)
        c1.metric("HydraDB avg", f"{sum(h_lats)/len(h_lats):.0f}ms")
        c2.metric("Vector DB avg", f"{sum(c_lats)/len(c_lats):.0f}ms")

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
