"""
🔬 Graph RAG Explorer — Learn RAG Step by Step

A visual, interactive app that shows you exactly how
Retrieval Augmented Generation works with a Neo4j knowledge graph.
"""

import os
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

from rag.chunker import load_document, chunk_text, extract_entities
from rag.embedder import embed_text, embed_texts
from rag.graph_store import GraphStore
from rag.retriever import retrieve
from rag.generator import generate_answer

load_dotenv()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Graph RAG Explorer", layout="wide")

st.title("Graph RAG Explorer")
st.caption("Learn how Retrieval Augmented Generation works — step by step")

# ── Sidebar: Connection Settings ─────────────────────────────
with st.sidebar:
    st.header("Settings")
    neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = st.text_input("Neo4j User", value=os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")

    st.divider()
    st.markdown("""
    **How to run Neo4j locally:**
    ```bash
    docker run -d \\
      --name neo4j \\
      -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/password \\
      neo4j:latest
    ```
    Then set password above to `password`
    """)


# ── Helper: Connect to Neo4j ────────────────────────────────
@st.cache_resource
def get_graph(uri, user, password):
    return GraphStore(uri, user, password)


# ══════════════════════════════════════════════════════════════
#  THE RAG PIPELINE — 4 VISUAL STEPS
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
### The RAG Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. INGEST  │───►│  2. STORE   │───►│ 3. RETRIEVE │───►│ 4. GENERATE │
│  Load &     │    │  Build Neo4j│    │  Find chunks │    │  Ask Claude │
│  Chunk docs │    │  Graph      │    │  via graph   │    │  with context│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```
""")

# ── STEP 1: Document Loading & Chunking ──────────────────────
st.markdown("---")
st.header("Step 1: Load & Chunk Document")
st.markdown("""
> **What happens here:** We load a text document and split it into small chunks (paragraphs).
> Each chunk is a self-contained piece of knowledge that can be independently retrieved.
""")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    use_sample = st.checkbox("Use sample knowledge base", value=True)

    if use_sample:
        doc_text = load_document("sample_data/knowledge.txt")
    elif uploaded_file:
        doc_text = uploaded_file.read().decode("utf-8")
    else:
        doc_text = None

    if doc_text:
        st.text_area("Raw Document", doc_text, height=200, disabled=True)

with col2:
    if doc_text:
        chunks = chunk_text(doc_text, source="knowledge")
        st.success(f"Split into **{len(chunks)} chunks**")

        for chunk in chunks:
            entities = extract_entities(chunk)
            entity_tags = " ".join(f"`{e['name']}`" for e in entities)
            with st.expander(f"Chunk {chunk.index}: {chunk.text[:50]}..."):
                st.write(chunk.text)
                if entities:
                    st.markdown(f"**Entities found:** {entity_tags}")
    else:
        st.info("Upload a document or use the sample to see chunks.")
        chunks = []

# ── STEP 2: Build the Knowledge Graph ───────────────────────
st.markdown("---")
st.header("Step 2: Build Knowledge Graph in Neo4j")
st.markdown("""
> **What happens here:** Each chunk becomes a **node**. Entities (people, technologies) become nodes too.
> We create **relationships**: `MENTIONED_IN`, `NEXT`, `CO_OCCURS` — this is the *graph* in Graph RAG.
""")

if doc_text and chunks:
    if st.button("Build Graph", type="primary"):
        try:
            graph = get_graph(neo4j_uri, neo4j_user, neo4j_password)

            with st.status("Building knowledge graph...", expanded=True) as status:
                # Clear existing data
                st.write("Clearing old graph data...")
                graph.clear()

                # Embed all chunks
                st.write("Generating embeddings for chunks...")
                texts = [c.text for c in chunks]
                embeddings = embed_texts(texts)

                # Store chunks as nodes
                st.write("Creating Chunk nodes...")
                for chunk, emb in zip(chunks, embeddings):
                    graph.store_chunk(chunk, emb)

                # Link sequential chunks
                st.write("Linking sequential chunks (NEXT relationships)...")
                for i in range(len(chunks) - 1):
                    graph.link_sequential_chunks(chunks[i].id, chunks[i + 1].id)

                # Extract entities and create entity nodes + edges
                st.write("Extracting entities and building entity graph...")
                for chunk in chunks:
                    entities = extract_entities(chunk)
                    entity_names = []
                    for ent in entities:
                        graph.store_entity(ent["name"], ent["type"])
                        graph.link_entity_to_chunk(ent["name"], chunk.id)
                        entity_names.append(ent["name"])

                    # Link co-occurring entities
                    for i, name1 in enumerate(entity_names):
                        for name2 in entity_names[i + 1 :]:
                            graph.link_related_entities(name1, name2)

                status.update(label="Knowledge graph built!", state="complete")

            st.session_state["graph_built"] = True
            st.session_state["graph"] = graph

            # Visualize the graph
            st.subheader("Your Knowledge Graph")
            graph_data = graph.get_all_nodes_and_edges()

            # Build agraph nodes and edges
            ag_nodes = []
            ag_edges = []

            for node in graph_data["nodes"]:
                label = node["labels"][0]
                props = node["props"]
                if label == "Chunk":
                    ag_nodes.append(Node(
                        id=props["id"],
                        label=f"Chunk {props['index']}",
                        size=25,
                        color="#4CAF50",
                        title=props["text"][:100] + "...",
                    ))
                elif label == "Entity":
                    color = "#FF9800" if props.get("type") == "TECHNOLOGY" else "#2196F3"
                    ag_nodes.append(Node(
                        id=props["name"],
                        label=props["name"],
                        size=20,
                        color=color,
                        title=f"{props.get('type', 'ENTITY')}",
                    ))

            for edge in graph_data["edges"]:
                color_map = {
                    "MENTIONED_IN": "#999",
                    "NEXT": "#4CAF50",
                    "CO_OCCURS": "#FF9800",
                }
                ag_edges.append(Edge(
                    source=edge["from"],
                    target=edge["to"],
                    label=edge["type"],
                    color=color_map.get(edge["type"], "#999"),
                ))

            config = Config(
                width=900,
                height=500,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
            )

            agraph(nodes=ag_nodes, edges=ag_edges, config=config)

            st.markdown("""
            **Legend:**
            - 🟢 Green = Chunk nodes (pieces of your document)
            - 🟠 Orange = Technology entities
            - 🔵 Blue = Person/Org entities
            - Green lines = NEXT (document flow)
            - Gray lines = MENTIONED_IN
            - Orange lines = CO_OCCURS
            """)

        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            st.info("Make sure Neo4j is running. See sidebar for Docker instructions.")
else:
    st.info("Load a document in Step 1 first.")


# ── STEP 3 & 4: Query, Retrieve, Generate ───────────────────
st.markdown("---")
st.header("Step 3 & 4: Ask a Question (Retrieve + Generate)")
st.markdown("""
> **What happens here:**
> 1. Your question is converted to a vector embedding
> 2. We find the most similar chunks via **vector search**
> 3. We **traverse the graph** to find additional related context
> 4. All context is sent to **Claude** which generates a grounded answer
""")

question = st.text_input(
    "Ask a question about the knowledge base:",
    placeholder="e.g., What is RAG and why is it useful?",
)

if question and st.session_state.get("graph_built"):
    graph = st.session_state["graph"]

    with st.status("Running RAG pipeline...", expanded=True) as status:
        # Step 3a: Embed the question
        st.write("**3a.** Embedding your question...")
        query_embedding = embed_text(question)
        st.code(f"Vector: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ... ] ({len(query_embedding)} dims)")

        # Step 3b: Retrieve
        st.write("**3b.** Searching knowledge graph...")
        results = retrieve(graph, query_embedding, top_k=2)

        # Show retrieval log
        for log_entry in results["traversal_log"]:
            st.write(f"  **{log_entry['step']}:** {log_entry['detail']}")
            for item in log_entry["data"]:
                st.json(item)

        # Step 4: Generate
        if anthropic_key:
            st.write("**4.** Sending context to Claude...")
            gen_result = generate_answer(question, results["all_context"], anthropic_key)
            status.update(label="Done!", state="complete")
        else:
            gen_result = None
            status.update(label="Retrieval complete (no API key for generation)", state="complete")

    # ── Display Results ──────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Retrieved Context")

        st.markdown("**From Vector Search** (cosine similarity):")
        for r in results["vector_results"]:
            score = r.get("score", 0)
            st.markdown(f"- `{r['id']}` (score: {score:.3f})")
            st.caption(r["text"][:150] + "...")

        if results["graph_results"]:
            st.markdown("**From Graph Traversal** (connected nodes):")
            for r in results["graph_results"]:
                st.markdown(f"- `{r['id']}` (via {r['source']})")
                st.caption(r["text"][:150] + "...")
        else:
            st.caption("No additional context from graph traversal.")

    with col_right:
        st.subheader("Generated Answer")
        if gen_result:
            st.markdown(gen_result["answer"])

            with st.expander("See full prompt sent to Claude"):
                st.code(gen_result["prompt"], language="text")
        else:
            st.warning("Add your Anthropic API key in the sidebar to enable answer generation.")
            st.markdown("**Context that would be sent to the LLM:**")
            st.text_area("Context", results["all_context"], height=300, disabled=True)

elif question and not st.session_state.get("graph_built"):
    st.warning("Build the knowledge graph in Step 2 first!")


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### How Graph RAG Differs from Plain RAG

| | **Plain RAG** | **Graph RAG** |
|---|---|---|
| **Storage** | Flat vector store | Graph with nodes & relationships |
| **Retrieval** | Vector similarity only | Vector search + graph traversal |
| **Context** | Isolated chunks | Connected, contextual chunks |
| **Strength** | Simple, fast | Rich context, better for complex queries |

Built with Neo4j + Claude + Streamlit
""")
