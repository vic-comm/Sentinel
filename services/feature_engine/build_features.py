import pandas as pd
import numpy as np
import networkx as nx
import torch

from sklearn.cluster import KMeans
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

Path("data/training").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_all():
    sim = pd.read_json("sentinel_training_data.jsonl", lines=True)

    try:
        elliptic = pd.read_json("data/processed/elliptic_cross_modal.jsonl", lines=True)
    except:
        elliptic = pd.DataFrame()

    try:
        alchemy = pd.read_parquet("data/processed/alchemy_transactions.parquet")
    except:
        alchemy = pd.DataFrame()

    df = pd.concat([sim, elliptic, alchemy], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df

# ─────────────────────────────────────────────
# IDENTITY RESOLUTION (probabilistic)
# ─────────────────────────────────────────────

def build_entity_cluster(df):
    df["entity_base"] = (
        df["sender_hash"]
        .fillna(df["sender_wallet_hash"])
        .fillna("unknown")
    )

    df["amount_bucket"] = (df["amount_usd"] // 100).fillna(0)
    df["hour"] = df["timestamp"].dt.hour.fillna(0)

    df["entity_cluster"] = (
        df["entity_base"].astype(str).str[:6] +
        "_" + df["amount_bucket"].astype(str) +
        "_" + (df["hour"] // 6).astype(str)
    )

    return df

# ─────────────────────────────────────────────
# TEMPORAL FEATURES
# ─────────────────────────────────────────────

def add_temporal_features(df):
    df = df.sort_values("timestamp").copy()
    df["entity"] = df["entity_cluster"]

    df["txn_count_24h"] = (
        df.groupby("entity")
          .rolling("24h", on="timestamp")["transaction_id"]
          .count()
          .reset_index(level=0, drop=True)
    )

    df["amount_mean_50"] = (
        df.groupby("entity")["amount_usd"]
          .rolling(50)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["amount_drift"] = df["amount_usd"] - df["amount_mean_50"]

    return df

# ─────────────────────────────────────────────
# GRAPH BUILDING
# ─────────────────────────────────────────────

def build_graph(df):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        s = row.get("sender_hash") or row.get("sender_wallet_hash")
        r = row.get("receiver_hash") or row.get("receiver_wallet_hash")

        if s and r:
            G.add_edge(s, r, weight=row.get("amount_usd", 0))

    return G

# ─────────────────────────────────────────────
# GRAPH FEATURES
# ─────────────────────────────────────────────

def add_graph_features(df):
    G = build_graph(df)

    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G.to_undirected())

    df["node"] = df["sender_hash"].fillna(df["sender_wallet_hash"])

    df["pagerank"] = df["node"].map(pagerank).fillna(0)
    df["clustering"] = df["node"].map(clustering).fillna(0)

    return df, G

# ─────────────────────────────────────────────
# BEHAVIORAL FEATURES
# ─────────────────────────────────────────────

def add_behavioral_clusters(df):
    X = df[["amount_usd", "txn_count_24h"]].fillna(0)

    kmeans = KMeans(n_clusters=8, random_state=42)
    df["behavior_cluster"] = kmeans.fit_predict(X)

    cluster_risk = df.groupby("behavior_cluster")["is_fraud"].mean()
    df["cluster_risk"] = df["behavior_cluster"].map(cluster_risk)

    return df

# ─────────────────────────────────────────────
# UNCERTAINTY FEATURES
# ─────────────────────────────────────────────

def add_uncertainty(df):
    df["source_weight"] = df["source"].map({
        "simulator": 0.3,
        "elliptic_real": 1.0,
        "alchemy_real": 0.8
    }).fillna(0.5)

    df["label_confidence"] = df["source"].map({
        "simulator": 0.6,
        "elliptic_real": 0.95,
        "alchemy_real": 0.7
    }).fillna(0.5)

    return df

# ─────────────────────────────────────────────
# GRAPH → PYG DATA
# ─────────────────────────────────────────────

def build_pyg_data(G, df):
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # node features
    node_features = []

    for n in nodes:
        subset = df[df["node"] == n]
        if len(subset) == 0:
            node_features.append([0,0,0])
        else:
            node_features.append([
                subset["amount_usd"].mean(),
                subset["txn_count_24h"].mean(),
                subset["pagerank"].mean()
            ])

    x = torch.tensor(node_features, dtype=torch.float)

    # labels (weak supervision)
    y = []
    for n in nodes:
        subset = df[df["node"] == n]
        if len(subset) == 0:
            y.append(0)
        else:
            y.append(int(subset["is_fraud"].mean() > 0.5))

    y = torch.tensor(y, dtype=torch.long)


def main():
    print("Loading data...")
    df = load_all()

    print("Building identity clusters...")
    df = build_entity_cluster(df)

    print("Temporal features...")
    df = add_temporal_features(df)

    print("Graph features...")
    df, G = add_graph_features(df)

    print("Behavioral features...")
    df = add_behavioral_clusters(df)

    print("Uncertainty features...")
    df = add_uncertainty(df)

    print("Building PyG graph...")
    pyg_data, node_idx = build_pyg_data(G, df)


