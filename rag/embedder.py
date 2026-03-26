"""
╔══════════════════════════════════════════════════╗
║  Embedder — Converting Text to Vectors           ║
║                                                  ║
║  "Neo4j is a graph database"                     ║
║       │                                          ║
║       ▼                                          ║
║  [0.12, -0.45, 0.78, ..., 0.33]  (384 dims)     ║
║                                                  ║
║  Similar meanings → similar vectors              ║
║  This enables SEMANTIC search, not just keywords ║
╚══════════════════════════════════════════════════╝
"""

from sentence_transformers import SentenceTransformer

# Using a small, fast model — good enough for learning!
# all-MiniLM-L6-v2: 384 dimensions, very fast
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloads ~80MB first time)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    """Convert a single text string into a vector embedding."""
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple texts into vector embeddings (batched for speed)."""
    model = get_model()
    embeddings = model.encode(texts)
    return [e.tolist() for e in embeddings]
