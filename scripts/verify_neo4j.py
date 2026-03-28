"""
scripts/verify_neo4j.py
=======================
Runs a set of Cypher verification queries after load_neo4j.py completes.
Each query checks a specific fraud pattern is detectable in the graph.

If any check fails, it means the load or the simulator has a bug — stop
and fix before proceeding to GNN training.

Usage:
    python scripts/verify_neo4j.py
"""

from neo4j import GraphDatabase
import sys

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "sentinel_neo4j"

CHECKS = [
    # ── Basic counts ──────────────────────────────────────────────────────────
    {
        "name": "Identity nodes exist",
        "query": "MATCH (n:Identity) RETURN count(n) AS c",
        "assert": lambda r: r["c"] > 1000,
        "message": "Too few Identity nodes — did load_neo4j.py finish?",
    },
    {
        "name": "SENT_TO edges exist",
        "query": "MATCH ()-[r:SENT_TO]->() RETURN count(r) AS c",
        "assert": lambda r: r["c"] > 10_000,
        "message": "Too few edges — check load_neo4j.py output.",
    },

    # ── Fraud labels ──────────────────────────────────────────────────────────
    {
        "name": "Fraud edges exist",
        "query": "MATCH ()-[r:SENT_TO]->() WHERE r.`is_fraud:int` = 1 RETURN count(r) AS c",
        "assert": lambda r: r["c"] > 0,
        "message": "No fraud edges found — is_fraud may not be loading correctly.",
    },
    {
        "name": "Fraud rate is reasonable (1–10%)",
        "query": """
            MATCH ()-[r:SENT_TO]->()
            RETURN toFloat(sum(CASE WHEN r.`is_fraud:int` = 1 THEN 1 ELSE 0 END)) / count(r) AS rate
        """,
        "assert": lambda r: 0.005 < r["rate"] < 0.15,
        "message": "Fraud rate outside expected range. Check FIAT_FRAUD_RATE in simulator.",
    },
    # ── Cross-modal detection ─────────────────────────────────────────────────
    {
        "name": "Cross-modal fraud pairs exist",
        "query": """
            MATCH ()-[r:SENT_TO]->()
            WHERE r.cross_modality_fraud_id IS NOT NULL
            RETURN count(DISTINCT r.cross_modality_fraud_id) AS c
        """,
        "assert": lambda r: r["c"] > 10,
        "message": "No cross-modal fraud sequences. "
                   "Check that CryptoEx simulator received bridges from NeoBank.",
    },
    {
        "name": "Each cross-modal sequence has BOTH fiat and crypto legs",
        "query": """
            MATCH ()-[r:SENT_TO]->()
            WHERE r.cross_modality_fraud_id IS NOT NULL
            WITH r.cross_modality_fraud_id AS fraud_id,
                 collect(DISTINCT r.modality) AS modalities
            WHERE size(modalities) > 1
            RETURN count(*) AS c
        """,
        "assert": lambda r: r["c"] > 0,
        "message": "Cross-modal pairs have only one leg — the linking by identity_hash "
                   "is broken. Verify SHARED_SALT is identical in both simulators.",
    },

    # ── Fraud pattern topology ────────────────────────────────────────────────
    {
        "name": "Mule networks have fan-in topology (hub with 3+ inbound fraud edges)",
        "query": """
            MATCH (hub:Identity)<-[r:SENT_TO {fraud_type: "mule_network"}]-()
            WITH hub, count(r) AS in_degree
            WHERE in_degree >= 3
            RETURN count(hub) AS c
        """,
        "assert": lambda r: r["c"] > 0,
        "message": "No mule network hubs found. "
                   "Check _inject_mule_network() in the simulator.",
    },
    {
        "name": "Account takeover edges exist",
        "query": """
            MATCH ()-[r:SENT_TO {fraud_type: "account_takeover"}]->()
            RETURN count(r) AS c
        """,
        "assert": lambda r: r["c"] > 0,
        "message": "No ATO edges. Check _inject_ato() in the simulator.",
    },

    # ── Identity is reachable via multiple modalities ─────────────────────────
     {
        "name": "Some Identity nodes appear in both fiat and crypto edges",
        "query": """
            MATCH (n:Identity)-[r:SENT_TO]->()
            WITH n, collect(DISTINCT r.modality) AS mods
            WHERE 'fiat' IN mods AND 'crypto' IN mods
            RETURN count(n) AS c
        """,
        "assert": lambda r: r["c"] > 0,
        "message": "No identity appears in both fiat and crypto — "
                   "cross-institutional matching is broken.",
    },
]


def run_checks(driver):
    passed = 0
    failed = 0
    results = []

    print(f"\nRunning {len(CHECKS)} verification checks...\n")

    with driver.session(database="neo4j") as session:
        for check in CHECKS:
            try:
                row    = session.run(check["query"]).single()
                ok     = check["assert"](row)
                status = "PASS" if ok else "FAIL"
                value  = dict(row) if row else {}
            except Exception as e:
                ok     = False
                status = "ERROR"
                value  = {"error": str(e)}

            results.append((status, check["name"], value, check.get("message", "")))

            icon = "✓" if ok else "✗"
            print(f"  {icon} [{status}] {check['name']}")
            if not ok:
                print(f"         Value:   {value}")
                print(f"         Issue:   {check['message']}")
                print()

            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\n{'─'*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(CHECKS)} checks")

    if failed == 0:
        print("\nAll checks passed. Neo4j graph is correctly populated.")
        print("Next: proceed to GraphSAGE training (Week 6).")
    else:
        print(f"\n{failed} check(s) failed. Fix the issues before training the GNN.")
        print("Common causes:")
        print("  - load_neo4j.py did not finish (run it again)")
        print("  - sentinel_simulator.py had errors during cross-modal bridge generation")
        print("  - SHARED_SALT mismatch between NeoBank and CryptoEx simulators")

    return failed == 0


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"Cannot connect to Neo4j: {e}")
        print("Run: docker compose up -d neo4j")
        sys.exit(1)

    try:
        ok = run_checks(driver)
        sys.exit(0 if ok else 1)
    finally:
        driver.close()