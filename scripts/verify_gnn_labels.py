# scripts/verify_gnn_labels.py
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "sentinel_neo4j"))

with driver.session(database="neo4j") as session:
    # Check what properties actually exist on edges
    r = session.run("MATCH ()-[r:SENT_TO]->() RETURN keys(r) AS props LIMIT 1").single()
    print(f"Edge properties: {r['props']}")

    # Check is_fraud:int values
    r = session.run("""
        MATCH ()-[r:SENT_TO]->()
        RETURN r.`is_fraud:int` AS f, count(*) AS c
        LIMIT 5
    """)
    print("is_fraud:int distribution:")
    for row in r:
        print(f"  {row['f']} → {row['c']:,}")

    # Check what the current edge query actually returns
    r = session.run("""
        MATCH ()-[r:SENT_TO]->()
        RETURN coalesce(toInteger(r.is_fraud), 0) AS f, count(*) AS c
    """)
    print("r.is_fraud (old query) distribution:")
    for row in r:
        print(f"  {row['f']} → {row['c']:,}")

driver.close()