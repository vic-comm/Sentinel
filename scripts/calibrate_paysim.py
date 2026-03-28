"""
scripts/calibrate_paysim.py
writes data/calibration/calibration.yaml.
"""
import pandas as pd
import numpy as np
import yaml
from scipy import stats
from pathlib import Path

print("Loading PaySim...")
df = pd.read_csv("data/raw/paysim.csv")

# Focus on TRANSFER type — closest to ACH/wire
transfers = df[df["type"] == "TRANSFER"]
legit = transfers[transfers["isFraud"] == 0]
fraud = transfers[transfers["isFraud"] == 1]

print(f"Legitimate transfers: {len(legit):,}")
print(f"Fraudulent transfers: {len(fraud):,}")

# ── Amount distribution ──────────────────────────────────
log_amounts = np.log(legit["amount"].clip(lower=1))
mu    = float(log_amounts.mean())
sigma = float(log_amounts.std())
print(f"\nLog-normal fit: mu={mu:.3f}, sigma={sigma:.3f}")
print(f"This implies median amount: ${np.exp(mu):,.0f}")

# PaySim is African mobile money so amounts are large.
# We scale down by subtracting from mu for each archetype.
# exp(mu - 4.7) ≈ $1,500 which is right for salary workers.

# ── Fraud amount multiplier ──────────────────────────────
fraud_median = float(fraud["amount"].median())
legit_median = float(legit["amount"].median())
fraud_multiplier = fraud_median / legit_median
print(f"\nFraud amount multiplier: {fraud_multiplier:.2f}x")

# ── Balance drain ratio ──────────────────────────────────
has_balance = fraud[fraud["oldbalanceOrg"] > 0].copy()
has_balance["drain"] = (
    has_balance["oldbalanceOrg"] - has_balance["newbalanceOrig"]
) / has_balance["oldbalanceOrg"]
drain_median = float(has_balance["drain"].median())
drain_p10    = float(has_balance["drain"].quantile(0.10))
print(f"Drain ratio — median: {drain_median:.3f}, p10: {drain_p10:.3f}")

# ── Fraud velocity ───────────────────────────────────────
fraud_vel = (fraud
    .groupby("nameOrig")["step"]
    .agg(["min", "max", "count"])
    .copy())
fraud_vel["span"] = fraud_vel["max"] - fraud_vel["min"]
burst_count_min = int(fraud_vel["count"].quantile(0.10))
burst_count_max = int(fraud_vel["count"].quantile(0.90))
burst_span_mean = float(fraud_vel["span"].mean())
print(f"Fraud burst: {burst_count_min}–{burst_count_max} txns over ~{burst_span_mean:.1f} hours")

# ── Timing: fraud hour distribution ─────────────────────
fraud_hours = fraud["step"] % 24
hour_counts = fraud_hours.value_counts().sort_index()
total = hour_counts.sum()
fraud_hour_weights = (hour_counts / total).tolist()
print(f"Fraud peaks at hours: {hour_counts.nlargest(3).index.tolist()}")

# ── Write calibration.yaml ───────────────────────────────
calibration = {
    "paysim_fit": {
        "mu": mu,
        "sigma": sigma,
        "note": "raw PaySim log-normal fit; subtract archetype offset from mu",
    },
    "archetype_amount_lognorm": {
        # mu offset scales the median amount per archetype
        # salary_worker: exp(mu - 4.7) ≈ $1,500
        # adjust these if your simulate summary shows wrong medians
        "salary_worker":  {"mu_offset": -4.7, "sigma_scale": 0.70},
        "freelancer":     {"mu_offset": -4.2, "sigma_scale": 1.00},
        "small_business": {"mu_offset": -3.2, "sigma_scale": 1.20},
        "retiree":        {"mu_offset": -5.5, "sigma_scale": 0.50},
        "student":        {"mu_offset": -6.0, "sigma_scale": 0.60},
    },
    "fraud_amount_multiplier":   round(fraud_multiplier, 2),
    "fraud_drain_ratio_median":  round(drain_median, 3),
    "fraud_drain_ratio_min":     round(drain_p10, 3),
    "fraud_burst_count":         [burst_count_min, burst_count_max],
    "fraud_burst_span_hours":    round(burst_span_mean, 1),
    "fraud_hour_weights":        [round(w, 4) for w in fraud_hour_weights],
}

Path("data/calibration").mkdir(exist_ok=True)
with open("data/calibration/calibration.yaml", "w") as f:
    yaml.dump(calibration, f, default_flow_style=False)

print("\nSaved → data/calibration/calibration.yaml")
print("\nPaste these archetype mu values into sentinel_simulator.py:")
base_mu = mu
for arch, cfg in calibration["archetype_amount_lognorm"].items():
    final_mu    = base_mu + cfg["mu_offset"]
    final_sigma = sigma   * cfg["sigma_scale"]
    median_amt  = np.exp(final_mu)
    print(f"  {arch:<20} mu={final_mu:.2f}, sigma={final_sigma:.2f}  "
          f"→ median ${median_amt:,.0f}")