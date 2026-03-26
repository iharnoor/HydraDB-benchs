"""
╔══════════════════════════════════════════════════╗
║  STEP 1: Document Loading & Chunking             ║
║                                                  ║
║  Raw Text ──► Paragraphs ──► Chunks              ║
║                                                  ║
║  WHY? LLMs have limited context windows.         ║
║  We split docs into small, meaningful pieces     ║
║  so we can retrieve ONLY what's relevant.        ║
╚══════════════════════════════════════════════════╝
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of text with metadata about where it came from."""
    id: str
    text: str
    source: str
    index: int  # position in original document

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Chunk({self.id}: '{preview}')"


def load_document(file_path: str) -> str:
    """Load a text file and return its contents."""
    with open(file_path, "r") as f:
        return f.read()


def chunk_text(text: str, source: str = "document") -> list[Chunk]:
    """
    Split text into chunks by paragraph.

    This is the simplest chunking strategy:
    - Split on double newlines (paragraphs)
    - Each paragraph becomes one chunk

    More advanced strategies exist (sliding window, semantic chunking)
    but paragraphs work great for learning!
    """
    # Split into paragraphs, strip whitespace, drop empties
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    for i, para in enumerate(paragraphs):
        chunk = Chunk(
            id=f"{source}_chunk_{i}",
            text=para,
            source=source,
            index=i,
        )
        chunks.append(chunk)

    return chunks


def extract_entities(chunk: Chunk) -> list[dict]:
    """
    Extract simple entities from a chunk using pattern matching.

    In production, you'd use an NLP model (spaCy, etc.) or an LLM
    to extract entities. Here we use a simple approach:
    - Capitalized multi-word phrases → likely proper nouns / names
    - Known tech terms
    """
    text = chunk.text
    entities = []

    # Find capitalized phrases (2+ words) — likely named entities
    # e.g. "Guido van Rossum", "Neo4j Inc"
    proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s+(?:van\s+)?[A-Z][a-z]+)+", text)
    for name in proper_nouns:
        entities.append({"name": name, "type": "PERSON_OR_ORG"})

    # Find known tech keywords
    tech_terms = [
        "Python", "Neo4j", "RAG", "Cypher", "LLM", "Claude",
        "vector embeddings", "semantic search", "graph database",
        "knowledge graph", "sentence-transformers",
        "Retrieval Augmented Generation",
    ]
    for term in tech_terms:
        if term.lower() in text.lower():
            entities.append({"name": term, "type": "TECHNOLOGY"})

    # Deduplicate by name
    seen = set()
    unique = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique.append(e)

    return unique
