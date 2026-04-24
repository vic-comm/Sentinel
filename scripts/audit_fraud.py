# scripts/audit_fraud_leakage.py
import pandas as pd
import numpy as np

train = pd.read_parquet("data/training/train.parquet")
fraud = train[train["is_fraud"] == True]
legit = train[train["is_fraud"] == False]

# For each feature, compute separation score
# If a single feature perfectly separates fraud from legit, it's a simulator artifact
results = []
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["is_fraud", "source_confidence"]]

for col in numeric_cols:
    fraud_mean = fraud[col].mean()
    legit_mean = legit[col].mean()
    fraud_std  = fraud[col].std()
    legit_std  = legit[col].std()
    pooled_std = np.sqrt((fraud_std**2 + legit_std**2) / 2 + 1e-9)
    # Cohen's d — measures separation between fraud and legit distributions
    cohens_d   = abs(fraud_mean - legit_mean) / pooled_std
    results.append({
        "feature":    col,
        "cohens_d":   round(cohens_d, 3),
        "fraud_mean": round(fraud_mean, 3),
        "legit_mean": round(legit_mean, 3),
    })

df = pd.DataFrame(results).sort_values("cohens_d", ascending=False)
print("Features with Cohen's d > 1.0 (strong simulator artifacts):")
print(df[df["cohens_d"] > 1.0].to_string(index=False))
print(f"\nTotal strong artifacts: {(df['cohens_d'] > 1.0).sum()}")
print(f"Total moderate artifacts (d > 0.5): {(df['cohens_d'] > 0.5).sum()}")