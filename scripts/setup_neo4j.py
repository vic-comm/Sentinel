"""
scripts/setup_neo4j.py
======================
Run once after `docker compose up -d neo4j` and the healthcheck passes.

Creates:
  - Uniqueness constraints on Identity.hash and Wallet.address_hash
  - Indexes for common query patterns
  - Verifies the connection and prints server info

Usage:
    python scripts/setup_neo4j.py

Dependencies:
    pip install neo4j
"""

import sys
import time
from neo4j import GraphDatabase, exceptions

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "sentinel_neo4j"


def wait_for_neo4j(driver, max_wait_seconds=120):
    """
    Neo4j takes ~45 seconds to fully start. Poll until it responds.
    """
    print(f"Waiting for Neo4j at {NEO4J_URI} ...")
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        try:
            driver.verify_connectivity()
            print("  Neo4j is up.")
            return
        except exceptions.ServiceUnavailable:
            print("  Not ready yet, retrying in 5s ...")
            time.sleep(5)
    print(f"ERROR: Neo4j did not become available within {max_wait_seconds}s.")
    sys.exit(1)


def setup_schema(driver):
    """
    Creates constraints and indexes.
    Constraints enforce uniqueness and auto-create an index.
    Additional indexes speed up common query patterns.
    """
    statements = [

        # ── Constraints (uniqueness + implicit index) ─────────────────────
        """
        CREATE CONSTRAINT identity_hash_unique IF NOT EXISTS
        FOR (n:Identity) REQUIRE n.hash IS UNIQUE
        """,
        """
        CREATE CONSTRAINT wallet_hash_unique IF NOT EXISTS
        FOR (n:Wallet) REQUIRE n.address_hash IS UNIQUE
        """,

        # ── Indexes for common lookup patterns ────────────────────────────

        # Look up Identity by risk_score (fraud review queues)
        """
        CREATE INDEX identity_risk_score IF NOT EXISTS
        FOR (n:Identity) ON (n.risk_score)
        """,

        # Look up Identity by fraud label (training data queries)
        """
        CREATE INDEX identity_is_fraud IF NOT EXISTS
        FOR (n:Identity) ON (n.is_fraud)
        """,

        # Look up Identity by first_seen (new account detection)
        """
        CREATE INDEX identity_first_seen IF NOT EXISTS
        FOR (n:Identity) ON (n.first_seen)
        """,

        # Look up SENT_TO edges by fraud_type (pattern queries)
        """
        CREATE INDEX sent_to_fraud_type IF NOT EXISTS
        FOR ()-[r:SENT_TO]-() ON (r.fraud_type)
        """,

        # Look up SENT_TO edges by timestamp (time-range queries)
        """
        CREATE INDEX sent_to_timestamp IF NOT EXISTS
        FOR ()-[r:SENT_TO]-() ON (r.timestamp)
        """,

        # Look up SENT_TO edges by fraud_network_id (mule ring queries)
        """
        CREATE INDEX sent_to_network_id IF NOT EXISTS
        FOR ()-[r:SENT_TO]-() ON (r.fraud_network_id)
        """,

        # Look up SENT_TO edges by cross_modality_fraud_id
        """
        CREATE INDEX sent_to_cross_modal IF NOT EXISTS
        FOR ()-[r:SENT_TO]-() ON (r.cross_modality_fraud_id)
        """,
    ]

    print("\nCreating schema (constraints + indexes)...")
    with driver.session(database="neo4j") as session:
        for stmt in statements:
            clean = " ".join(stmt.split())   # collapse whitespace for display
            print(f"  {clean[:80]}...")
            session.run(stmt)

    print("Schema ready.")


def print_server_info(driver):
    with driver.session(database="neo4j") as session:
        result = session.run("CALL dbms.components() YIELD name, versions, edition")
        row = result.single()
        if row:
            print(f"\nNeo4j server: {row['name']} {row['versions'][0]} ({row['edition']})")

        result = session.run("""
            MATCH (n) RETURN count(n) AS nodes
        """)
        row = result.single()
        print(f"Current node count: {row['nodes']:,}")

        result = session.run("""
            MATCH ()-[r]->() RETURN count(r) AS edges
        """)
        row = result.single()
        print(f"Current edge count: {row['edges']:,}")


def print_useful_queries():
    print("""
─────────────────────────────────────────────────────────────
Useful Cypher queries (run in http://localhost:7474):
─────────────────────────────────────────────────────────────

// Count nodes and edges after loading
MATCH (n:Identity) RETURN count(n) AS identity_nodes;
MATCH ()-[r:SENT_TO]->() RETURN count(r) AS transactions;

// Find mule networks (aggregator with 5+ inbound fraud edges)
MATCH (hub:Identity)<-[r:SENT_TO {fraud_type: "mule_network"}]-()
WITH hub, count(r) AS in_degree
WHERE in_degree >= 5
RETURN hub.hash, in_degree
ORDER BY in_degree DESC
LIMIT 10;

// Find cross-modal fraud pairs
MATCH ()-[r:SENT_TO]->()
WHERE r.cross_modality_fraud_id IS NOT NULL
WITH r.cross_modality_fraud_id AS fraud_id,
     collect(DISTINCT r.modality) AS modalities,
     collect(DISTINCT r.amount_usd) AS amounts,
     min(r.timestamp) AS first_ts,
     max(r.timestamp) AS last_ts
WHERE size(modalities) > 1
RETURN fraud_id, modalities, amounts, first_ts, last_ts
LIMIT 10;

// Find shortest path from unknown node to known fraud
MATCH path = shortestPath(
    (unknown:Identity {hash: $target_hash})-[:SENT_TO*..5]-(fraud:Identity {is_fraud: true})
)
RETURN path, length(path)
ORDER BY length(path)
LIMIT 5;

// Visualise a mule ring (paste a fraud_network_id from your data)
MATCH (n)-[r:SENT_TO {fraud_network_id: $network_id}]->(m)
RETURN n, r, m;
─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        wait_for_neo4j(driver)
        print_server_info(driver)
        setup_schema(driver)
        print_server_info(driver)
        print_useful_queries()
        print("Neo4j setup complete. Run load_neo4j.py next to populate the graph.")
    finally:
        driver.close()
