"""
╔══════════════════════════════════════════════════════╗
║  STEP 2: Neo4j Graph Storage                         ║
║                                                      ║
║  Chunks & Entities become NODES in the graph.        ║
║  Connections between them become RELATIONSHIPS.      ║
║                                                      ║
║     [Entity: Python] ──MENTIONED_IN──► [Chunk 0]     ║
║     [Chunk 0] ──NEXT──► [Chunk 1]                    ║
║     [Entity: Neo4j] ──MENTIONED_IN──► [Chunk 1]      ║
║     [Entity: Neo4j] ──RELATED_TO──► [Entity: Cypher] ║
║                                                      ║
║  WHY a graph? Unlike flat vector stores, graphs      ║
║  capture HOW concepts connect — enabling richer      ║
║  retrieval by traversing relationships.              ║
╚══════════════════════════════════════════════════════╝
"""

from neo4j import GraphDatabase
from rag.chunker import Chunk


class GraphStore:
    """Manages the Neo4j knowledge graph for our RAG system."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for fast lookups."""
        with self.driver.session() as session:
            # Index on Chunk.id for fast retrieval
            session.run(
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
            )
            # Index on Entity.name for fast entity lookups
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )

    def clear(self):
        """Remove all nodes and relationships (fresh start)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def store_chunk(self, chunk: Chunk, embedding: list[float]):
        """
        Store a chunk as a node in Neo4j.

        Node label: Chunk
        Properties: id, text, source, index, embedding
        """
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text,
                    c.source = $source,
                    c.index = $index,
                    c.embedding = $embedding
                """,
                id=chunk.id,
                text=chunk.text,
                source=chunk.source,
                index=chunk.index,
                embedding=embedding,
            )

    def store_entity(self, name: str, entity_type: str):
        """
        Store an entity (person, technology, concept) as a node.

        Node label: Entity
        Properties: name, type
        """
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                """,
                name=name,
                type=entity_type,
            )

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str):
        """
        Create a MENTIONED_IN relationship: Entity ──► Chunk

        This means: "this entity appears in this chunk"
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {name: $entity_name})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                entity_name=entity_name,
                chunk_id=chunk_id,
            )

    def link_sequential_chunks(self, chunk_id_1: str, chunk_id_2: str):
        """
        Create a NEXT relationship: Chunk ──► Chunk

        Preserves document order so we can retrieve surrounding context.
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:Chunk {id: $id1})
                MATCH (c2:Chunk {id: $id2})
                MERGE (c1)-[:NEXT]->(c2)
                """,
                id1=chunk_id_1,
                id2=chunk_id_2,
            )

    def link_related_entities(self, name1: str, name2: str):
        """
        Create a CO_OCCURS relationship between entities
        that appear in the same chunk.

        Entity ──CO_OCCURS──► Entity
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e1:Entity {name: $name1})
                MATCH (e2:Entity {name: $name2})
                WHERE e1 <> e2
                MERGE (e1)-[:CO_OCCURS]-(e2)
                """,
                name1=name1,
                name2=name2,
            )

    def get_all_nodes_and_edges(self) -> dict:
        """Return all nodes and relationships for visualization."""
        with self.driver.session() as session:
            # Get all nodes
            nodes_result = session.run(
                """
                MATCH (n)
                RETURN n, labels(n) as labels, properties(n) as props
                """
            )
            nodes = []
            for record in nodes_result:
                props = dict(record["props"])
                props.pop("embedding", None)  # don't send huge vectors to UI
                nodes.append({
                    "labels": record["labels"],
                    "props": props,
                })

            # Get all relationships
            edges_result = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN properties(a) as from_props, labels(a) as from_labels,
                       type(r) as rel_type,
                       properties(b) as to_props, labels(b) as to_labels
                """
            )
            edges = []
            for record in edges_result:
                edges.append({
                    "from": record["from_props"].get("id") or record["from_props"].get("name"),
                    "to": record["to_props"].get("id") or record["to_props"].get("name"),
                    "type": record["rel_type"],
                })

            return {"nodes": nodes, "edges": edges}

    def close(self):
        self.driver.close()
