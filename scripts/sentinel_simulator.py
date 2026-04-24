# """
# Sentinel Training Data Simulator — v3.0 (Path B)
# =================================================
# Core change from v2: fraud is defined by SEQUENCE and GRAPH TOPOLOGY,
# not by individual transaction feature flags.

# What changed:
#   - ATO: normal amounts, normal hours, normal transfer types
#          Only signal: rapid sequence + device/IP change (probabilistic)
#   - Mule network: archetype-normal amounts at each stage
#                   Mule cut is variable (5-20%), some mules defect
#                   Stage 2 timing is irregular, not uniform
#   - Synthetic identity: realistic trust-building phase with real transactions
#                         Drain is partial, not 100%
#   - BEC: amount derived from business archetype, not fixed range
#   - All fraud hours: same distribution as legitimate (hour is not a signal)
#   - Transfer types: same distribution as legitimate across all fraud patterns

# What remains detectable (graph/sequence signals only):
#   - ATO: velocity_ratio spikes 10-15x, device_hash changes
#   - Mule: fan_in_ratio on aggregator, graph topology
#   - Synthetic: account_age_days < 90, velocity_ratio spike on day of drain
#   - BEC: receiver_in_degree = 0 (brand new mule account)
#   - Mixer: known_mixer_interaction, graph proximity
#   - Cross-modal: cross_modality_fraud_id links fiat→crypto
# """

# import hashlib
# import uuid
# import random
# import json
# import os
# import yaml
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from faker import Faker
# from tqdm import tqdm
# from collections import defaultdict
# from pathlib import Path

# fake = Faker()
# rng  = np.random.default_rng(42)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIGURATION
# # ─────────────────────────────────────────────────────────────────────────────

# SHARED_SALT   = os.getenv("SENTINEL_SALT", "sentinel_consortium_2026_v1")
# DEVICE_SALT   = os.getenv("DEVICE_SALT",   "sentinel_device_2026_v1")
# IP_SALT       = os.getenv("IP_SALT",       "sentinel_ip_2026_v1")

# START_TIME        = datetime(2026, 1, 1, 0, 0, 0)
# SIM_DAYS          = 30
# N_IDENTITIES      = 10_000
# TARGET_FIAT_TXN   = 1_500_000
# TARGET_CRYPTO_TXN = 500_000

# FIAT_FRAUD_RATE   = 0.020   # 2.0%
# CRYPTO_FRAUD_RATE = 0.030   # 3.0%
# CROSS_MODAL_RATE  = 0.20

# CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "data/calibration/calibration.yaml")


# # ─────────────────────────────────────────────────────────────────────────────
# # CALIBRATION
# # ─────────────────────────────────────────────────────────────────────────────

# def load_calibration(path):
#     p = Path(path)
#     if not p.exists():
#         print(f"[calibration] WARNING: {path} not found — using defaults.")
#         return {}
#     with open(p) as f:
#         cal = yaml.safe_load(f)
#     print(f"[calibration] Loaded from {path}")
#     return cal or {}


# def _build_archetype_lognorm(cal):
#     fit      = cal.get("paysim_fit", {})
#     base_mu  = fit.get("mu",   12.97)
#     base_sg  = fit.get("sigma", 1.30)
#     arch_cfg = cal.get("archetype_amount_lognorm", {})
#     defaults = {
#         "salary_worker":  {"mu_offset": -4.7, "sigma_scale": 0.70},
#         "freelancer":     {"mu_offset": -4.2, "sigma_scale": 1.00},
#         "small_business": {"mu_offset": -3.2, "sigma_scale": 1.20},
#         "retiree":        {"mu_offset": -5.5, "sigma_scale": 0.50},
#         "student":        {"mu_offset": -6.0, "sigma_scale": 0.60},
#     }
#     result = {}
#     for arch, fallback in defaults.items():
#         cfg   = arch_cfg.get(arch, fallback)
#         mu    = base_mu + cfg.get("mu_offset",   fallback["mu_offset"])
#         sigma = base_sg * cfg.get("sigma_scale", fallback["sigma_scale"])
#         result[arch] = (round(mu, 4), round(sigma, 4))
#     return result


# def _build_fraud_params(cal):
#     raw_burst = cal.get("fraud_burst_count", [3, 5])
#     return {
#         "burst_count": (max(3, raw_burst[0]), max(5, raw_burst[1])),
#     }


# print("[calibration] Loading...")
# _CAL               = load_calibration(CALIBRATION_PATH)
# _ARCHETYPE_LOGNORM = _build_archetype_lognorm(_CAL)
# _FRAUD_PARAMS      = _build_fraud_params(_CAL)
# print("[calibration] Ready.\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # ARCHETYPES
# # ─────────────────────────────────────────────────────────────────────────────

# ARCHETYPES = {
#     "salary_worker":  {"weight": 0.40, "txn_per_day": 2.5,  "income_range": (3000,  8000),  "counterparties": (5,  15), "repeat_rate": 0.80, "kyc_status": "verified"},
#     "freelancer":     {"weight": 0.25, "txn_per_day": 5.0,  "income_range": (2000,  15000), "counterparties": (10, 30), "repeat_rate": 0.60, "kyc_status": "verified"},
#     "small_business": {"weight": 0.15, "txn_per_day": 12.0, "income_range": (10000, 100000),"counterparties": (20, 50), "repeat_rate": 0.50, "kyc_status": "verified"},
#     "retiree":        {"weight": 0.10, "txn_per_day": 1.0,  "income_range": (1500,  4000),  "counterparties": (3,  8),  "repeat_rate": 0.90, "kyc_status": "verified"},
#     "student":        {"weight": 0.10, "txn_per_day": 1.5,  "income_range": (500,   2000),  "counterparties": (2,  6),  "repeat_rate": 0.75, "kyc_status": "pending"},
# }
# for _arch, _lognorm in _ARCHETYPE_LOGNORM.items():
#     if _arch in ARCHETYPES:
#         ARCHETYPES[_arch]["amount_lognorm"] = _lognorm

# CRYPTO_ARCHETYPES = {
#     "hodler":   {"weight": 0.30, "txn_per_day": 0.05, "amount_range": (500,   5000)},
#     "trader":   {"weight": 0.25, "txn_per_day": 1.5,  "amount_range": (1000,  20000)},
#     "defi":     {"weight": 0.20, "txn_per_day": 0.5,  "amount_range": (500,   10000)},
#     "merchant": {"weight": 0.15, "txn_per_day": 3.0,  "amount_range": (50,    5000)},
#     "p2p":      {"weight": 0.10, "txn_per_day": 0.4,  "amount_range": (20,    500)},
# }

# # Hourly weight — one distribution for ALL transactions (fraud and legit)
# HOURLY_WEIGHTS = [
#     0.05, 0.03, 0.03, 0.03, 0.04, 0.06,
#     0.10, 0.20, 0.40, 0.60, 0.80, 0.90,
#     0.85, 0.80, 0.90, 1.00, 0.90, 0.75,
#     0.55, 0.40, 0.30, 0.20, 0.12, 0.07,
# ]
# _hw_total      = sum(HOURLY_WEIGHTS)
# HOURLY_WEIGHTS = [w / _hw_total for w in HOURLY_WEIGHTS]

# MIXER_ADDRESSES = [
#     "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",
#     "0x8d12A197cB00D4747a1fe03395095ce2A5CC6819",
#     "0xA160cdAB225685dA1d56aa342Ad8841c3b53f291",
# ]


# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def sha256(v: str) -> str:
#     return hashlib.sha256(v.encode()).hexdigest()

# def identity_hash(email: str) -> str:
#     return sha256(f"{email}{SHARED_SALT}")

# def device_hash(d: str) -> str:
#     return sha256(f"{d}{DEVICE_SALT}")

# def ip_hash(ip: str) -> str:
#     return sha256(f"{ip}{IP_SALT}")

# def sample_hour() -> int:
#     return int(rng.choice(24, p=HOURLY_WEIGHTS))

# def rand_ts(day: int) -> datetime:
#     """Single hour distribution for ALL transactions. Fraud hour is not a signal."""
#     return START_TIME + timedelta(
#         days=day,
#         hours=sample_hour(),
#         minutes=random.randint(0, 59),
#         seconds=random.randint(0, 59),
#     )

# def sample_amount(archetype: str) -> float:
#     mu, sigma = ARCHETYPES[archetype]["amount_lognorm"]
#     return max(10.0, float(rng.lognormal(mu, sigma)))

# def sample_transfer_type() -> str:
#     """Single transfer type distribution for ALL transactions."""
#     return random.choices(
#         ["ach", "wire", "p2p", "rtp"],
#         weights=[0.60, 0.20, 0.15, 0.05]
#     )[0]

# def txn_id(prefix: str) -> str:
#     return f"{prefix}_{uuid.uuid4().hex[:12]}"

# def fake_wallet() -> str:
#     return f"0x{uuid.uuid4().hex[:40]}"

# def fake_tx_hash() -> str:
#     return f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:24]}"


# # ─────────────────────────────────────────────────────────────────────────────
# # IDENTITY POOL
# # ─────────────────────────────────────────────────────────────────────────────

# # def build_identity_pool(n: int) -> list[dict]:
# #     arch_keys    = list(ARCHETYPES.keys())
# #     arch_weights = [ARCHETYPES[a]["weight"] for a in arch_keys]

# #     identities = []
# #     print(f"Building {n:,} consortium identities...")
# #     for _ in tqdm(range(n)):
# #         email = fake.email()
# #         arch  = random.choices(arch_keys, weights=arch_weights)[0]
# #         cfg   = ARCHETYPES[arch]

# #         identities.append({
# #             "identity_hash":     identity_hash(email),
# #             "email":             email,
# #             "archetype":         arch,
# #             "income_monthly":    random.randint(*cfg["income_range"]),
# #             "account_age_days":  random.randint(30, 3650),
# #             "kyc_status":        cfg["kyc_status"],
# #             "velocity_baseline": cfg["txn_per_day"] * random.uniform(0.7, 1.3),
# #             "balance":           random.randint(*cfg["income_range"]) * random.uniform(1.0, 6.0),
# #             "counterparties":    [identity_hash(fake.email()) for _ in range(random.randint(*cfg["counterparties"]))],
# #             "device_hash":       device_hash(fake.uuid4()),
# #             "ip_hash":           ip_hash(fake.ipv4()),
# #             "has_crypto":        random.random() < 0.35,
# #             "crypto_wallet":     fake_wallet(),
# #             "crypto_archetype":  random.choices(
# #                 list(CRYPTO_ARCHETYPES.keys()),
# #                 weights=[v["weight"] for v in CRYPTO_ARCHETYPES.values()]
# #             )[0],
# #             "_txn_count_30d":  0,
# #             "_amount_sum_30d": 0.0,
# #             "_txn_history":    [],
# #         })
# #     return identities
# def build_identity_pool(n: int) -> list:
#     """
#     CHANGE: Add per-user financial profile flags that drive high-drain
#     legitimate transactions. These are the financial obligations that
#     create natural drain overlap with fraud:
#     - has_rent:     monthly rent payment (40-60% drain)
#     - has_mortgage: monthly mortgage (50-70% drain, but stable)
#     - has_recurring_large: tax, insurance, subscription bundles

#     These flags are used by _normal_txn to generate high-drain legitimate
#     transactions at realistic frequencies.
#     """
#     arch_keys    = list(ARCHETYPES.keys())
#     arch_weights = [ARCHETYPES[a]["weight"] for a in arch_keys]

#     identities = []
#     print(f"Building {n:,} consortium identities...")
#     for _ in tqdm(range(n)):
#         email = fake.email()
#         arch  = random.choices(arch_keys, weights=arch_weights)[0]
#         cfg   = ARCHETYPES[arch]

#         # Financial obligation profile — determines whether this user
#         # has recurring high-drain payments (overlap with fraud drain)
#         arch_rent_prob = {
#             "salary_worker":  0.65,   # most renters/mortgage holders
#             "freelancer":     0.70,
#             "small_business": 0.80,   # business rent
#             "retiree":        0.40,   # many own homes outright
#             "student":        0.55,
#         }

#         identities.append({
#             "identity_hash":     identity_hash(email),
#             "email":             email,
#             "archetype":         arch,
#             "income_monthly":    random.randint(*cfg["income_range"]),
#             "account_age_days":  random.randint(30, 3650),
#             "kyc_status":        cfg["kyc_status"],
#             "velocity_baseline": cfg["txn_per_day"] * random.uniform(0.7, 1.3),
#             "balance":           random.randint(*cfg["income_range"]) * random.uniform(1.0, 6.0),
#             "counterparties":    [identity_hash(fake.email()) for _ in range(random.randint(*cfg["counterparties"]))],
#             "device_hash":       device_hash(fake.uuid4()),
#             "ip_hash":           ip_hash(fake.ipv4()),
#             "has_crypto":        random.random() < 0.35,
#             "crypto_wallet":     fake_wallet(),
#             "crypto_archetype":  random.choices(
#                 list(CRYPTO_ARCHETYPES.keys()),
#                 weights=[v["weight"] for v in CRYPTO_ARCHETYPES.values()]
#             )[0],

#             # Financial obligations driving high-drain legitimate transactions
#             "has_large_recurring": random.random() < arch_rent_prob.get(arch, 0.50),
#             "recurring_drain_pct": random.uniform(0.35, 0.65),  # what % of balance this costs
#             "recurring_receiver":  identity_hash(fake.email()),  # stable counterparty (landlord etc.)

#             "_txn_count_30d":  0,
#             "_amount_sum_30d": 0.0,
#             "_txn_history":    [],
#         })
#     return identities



# # ─────────────────────────────────────────────────────────────────────────────
# # VELOCITY FEATURES
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_velocity_features(user: dict, current_ts: datetime, windows: list[int]) -> dict:
#     feats   = {}
#     history = user["_txn_history"]
#     for h in windows:
#         cutoff = current_ts - timedelta(hours=h)
#         w      = [(ts, amt) for ts, amt in history if ts >= cutoff]
#         feats[f"count_{h}h"]      = len(w)
#         feats[f"sum_amount_{h}h"] = round(sum(a for _, a in w), 2)
#     return feats

# def update_user_history(user: dict, ts: datetime, amount: float):
#     user["_txn_history"].append((ts, amount))
#     cutoff = ts - timedelta(days=7)
#     user["_txn_history"] = [(t, a) for t, a in user["_txn_history"] if t >= cutoff]
#     user["_txn_count_30d"]  += 1
#     user["_amount_sum_30d"] += amount

# def velocity_ratio(user: dict, count_24h: int) -> float:
#     baseline = user["velocity_baseline"]
#     return round(count_24h / baseline, 3) if baseline > 0 else 0.0

# def amount_ratio(user: dict, amount: float) -> float:
#     if user["_txn_count_30d"] == 0:
#         return 1.0
#     avg = user["_amount_sum_30d"] / user["_txn_count_30d"]
#     return round(amount / avg, 3) if avg > 0 else 1.0


# # ─────────────────────────────────────────────────────────────────────────────
# # TRANSACTION BUILDERS
# # ─────────────────────────────────────────────────────────────────────────────

# def build_fiat_txn(
#     sender, receiver_hash, amount, ts,
#     transfer_type="ach", transfer_network="domestic",
#     is_fraud=False, fraud_type=None,
#     fraud_network_id=None, cross_modality_fraud_id=None,
#     override_device_hash=None, override_ip_hash=None,
# ) -> dict:
#     balance_before = max(0.0, sender["balance"])
#     balance_after  = max(0.0, balance_before - amount)
#     drain_ratio    = round(amount / balance_before, 4) if balance_before > 0 else 0.0

#     vel = compute_velocity_features(sender, ts, [1, 6, 24, 168])
#     sender["balance"] = balance_after

#     is_first_time = receiver_hash not in sender["counterparties"]
#     if not is_first_time:
#         sender["counterparties"].append(receiver_hash)

#     update_user_history(sender, ts, amount)

#     return {
#         "transaction_id":          txn_id("neobank"),
#         "client_id":               "neobank_prod",
#         "timestamp":               ts.isoformat() + "Z",
#         "modality":                "fiat",
#         "identity_hash":           sender["identity_hash"],
#         "sender_hash":             sender["identity_hash"],
#         "receiver_hash":           receiver_hash,
#         "sender_device_hash":      override_device_hash or sender["device_hash"],
#         "sender_ip_hash":          override_ip_hash     or sender["ip_hash"],
#         "amount_usd":              round(amount, 2),
#         "transfer_type":           transfer_type,
#         "transfer_network":        transfer_network,
#         "sender_account_age_days": sender["account_age_days"],
#         "sender_kyc_status":       sender["kyc_status"],
#         "sender_archetype":        sender["archetype"],
#         "sender_lifetime_txn_count": sender["_txn_count_30d"],
#         "sender_lifetime_volume_usd": round(sender["_amount_sum_30d"], 2),
#         "is_first_time_receiver":  is_first_time,
#         "sender_balance_before":   round(balance_before, 2),
#         "sender_balance_after":    round(balance_after, 2),
#         "balance_drain_ratio":     drain_ratio,
#         "count_1h":                vel["count_1h"],
#         "count_6h":                vel["count_6h"],
#         "count_24h":               vel["count_24h"],
#         "count_7d":                vel["count_168h"],
#         "sum_amount_1h":           vel["sum_amount_1h"],
#         "sum_amount_24h":          vel["sum_amount_24h"],
#         "sum_amount_7d":           vel["sum_amount_168h"],
#         "velocity_ratio":          velocity_ratio(sender, vel["count_24h"]),
#         "amount_ratio":            amount_ratio(sender, amount),
#         "velocity_baseline_daily": round(sender["velocity_baseline"], 3),
#         "hour_of_day":             ts.hour,
#         "day_of_week":             ts.weekday(),
#         "is_weekend":              ts.weekday() >= 5,
#         "is_unusual_hour":         ts.hour < 6 or ts.hour >= 23,
#         "is_fraud":                is_fraud,
#         "fraud_type":              fraud_type,
#         "fraud_network_id":        fraud_network_id,
#         "cross_modality_fraud_id": cross_modality_fraud_id,
#     }


# def build_crypto_txn(
#     user, sender_wallet_hash, receiver_wallet_hash, amount_usd, ts,
#     transfer_type="withdrawal", receiver_wallet_type="user_wallet",
#     known_mixer_interaction=False, receiver_wallet_age_days=None,
#     is_fraud=False, fraud_type=None,
#     fraud_network_id=None, cross_modality_fraud_id=None,
#     override_device_hash=None, override_ip_hash=None,
# ) -> dict:
#     cryptocurrency = random.choices(
#         ["ETH", "BTC", "USDT", "USDC", "SOL"],
#         weights=[0.40, 0.25, 0.20, 0.10, 0.05]
#     )[0]

#     return {
#         "transaction_id":            txn_id("cryptoex"),
#         "client_id":                 "cryptoex_prod",
#         "timestamp":                 ts.isoformat() + "Z",
#         "modality":                  "crypto",
#         "identity_hash":             user["identity_hash"],
#         "sender_wallet_hash":        sender_wallet_hash,
#         "receiver_wallet_hash":      receiver_wallet_hash,
#         "user_email_hash":           user["identity_hash"],
#         "sender_device_hash":        override_device_hash or user["device_hash"],
#         "sender_ip_hash":            override_ip_hash     or user["ip_hash"],
#         "amount_usd":                round(amount_usd, 2),
#         "amount_crypto":             round(amount_usd / 2800.0, 6),
#         "cryptocurrency":            cryptocurrency,
#         "blockchain":                "ethereum",
#         "transaction_hash":          fake_tx_hash(),
#         "gas_price_gwei":            random.randint(10, 150),
#         "sender_wallet_age_days":    user.get("account_age_days", 30),
#         "sender_lifetime_txn_count": user.get("_txn_count_30d", 0),
#         "sender_lifetime_volume_usd": round(user.get("_amount_sum_30d", 0), 2),
#         "receiver_wallet_type":      receiver_wallet_type,
#         "receiver_wallet_age_days":  receiver_wallet_age_days if receiver_wallet_age_days is not None else random.randint(30, 1000),
#         "known_mixer_interaction":   known_mixer_interaction,
#         "is_first_time_receiver":    random.random() < 0.3,
#         "immediate_withdrawal":      False,
#         "hour_of_day":               ts.hour,
#         "day_of_week":               ts.weekday(),
#         "is_weekend":                ts.weekday() >= 5,
#         "is_unusual_hour":           ts.hour < 6 or ts.hour >= 23,
#         "is_fraud":                  is_fraud,
#         "fraud_type":                fraud_type,
#         "fraud_network_id":          fraud_network_id,
#         "cross_modality_fraud_id":   cross_modality_fraud_id,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # NEOBANK SIMULATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class NeoBankSimulator:
#     def __init__(self, identities):
#         self.identities           = identities
#         self.transactions         = []
#         self.cross_modal_bridges  = []

#     def _normal_txn(self, user, day):
#         amount        = sample_amount(user["archetype"])
#         repeat        = random.random() < ARCHETYPES[user["archetype"]]["repeat_rate"]
#         receiver      = random.choice(user["counterparties"]) if repeat and user["counterparties"] else identity_hash(fake.email())
#         transfer_type = sample_transfer_type()
#         transfer_network = random.choices(
#             ["same_bank", "domestic", "international"],
#             weights=[0.40, 0.50, 0.10]
#         )[0]
#         return build_fiat_txn(user, receiver, amount, rand_ts(day),
#                               transfer_type=transfer_type,
#                               transfer_network=transfer_network)

#     def _micro_anomaly(self, user, day):
#         """Legitimate large-but-real transfers — travel, inheritance, tax payment."""
#         kind   = random.choices(["travel_wire", "large_one_time", "income_spike"], weights=[0.50, 0.30, 0.20])[0]
#         if kind == "travel_wire":
#             amount, tt, tn = random.uniform(1000, 8000), "wire", "international"
#         elif kind == "large_one_time":
#             amount, tt, tn = random.uniform(10000, 50000), "wire", "domestic"
#         else:
#             amount, tt, tn = user["income_monthly"] * random.uniform(1.5, 3.0), "ach", "domestic"
#         return build_fiat_txn(user, identity_hash(fake.email()), amount, rand_ts(day),
#                               transfer_type=tt, transfer_network=tn)

#     # ── Fraud Pattern 1: Account Takeover ────────────────────────────────────
#     # Detectable by: device_hash change + velocity spike (not amount, not hour)

#     # def _inject_ato(self, victim, day) -> list:
#     #     """
#     #     PATH B CHANGES:
#     #       - Same hour distribution as legitimate (fraud_hour removed)
#     #       - Same transfer type distribution as legitimate
#     #       - Amounts drawn from victim's archetype lognormal (not balance drain)
#     #       - Device/IP change is probabilistic (80% of ATO cases), not always
#     #       - Burst interval is tighter (1-3 min) to create velocity spike

#     #     What makes this detectable:
#     #       velocity_ratio: 3-5 txns in 4-12 min vs baseline of 2.5/day → ratio 10-15x
#     #       device_hash: changes mid-session in 80% of cases
#     #       ip_hash: changes in 80% of cases
#     #       sequence_gap: inter-transaction time is 1-3 min instead of hours
#     #     """
#     #     # Device/IP change is probabilistic — real ATO uses stolen sessions too
#     #     attacker_took_session = random.random() < 0.20   # 20%: attacker has full session
#     #     if attacker_took_session:
#     #         # Session hijack: same device/IP, only velocity is the signal
#     #         override_device = None
#     #         override_ip     = None
#     #     else:
#     #         override_device = device_hash(fake.uuid4())
#     #         override_ip     = ip_hash(fake.ipv4_private())

#     #     # ATO happens at any hour — attacker knows the victim's timezone
#     #     base_ts = rand_ts(day)

#     #     txns         = []
#     #     burst_min, burst_max = _FRAUD_PARAMS["burst_count"]
#     #     n_transfers  = random.randint(burst_min, burst_max)

#     #     for i in range(n_transfers):
#     #         # Normal amount from victim's archetype — not balance drain
#     #         # The SEQUENCE is the signal, not the individual amount
#     #         amount = sample_amount(victim["archetype"])

#     #         # Tight timing: 1-3 minutes between transfers (velocity signal)
#     #         ts     = base_ts + timedelta(minutes=i * random.randint(1, 3))

#     #         # Normal transfer type — same distribution as legitimate
#     #         tt = sample_transfer_type()

#     #         # Receivers are always new (mules) — is_first_time_receiver = True
#     #         receiver = identity_hash(fake.email())

#     #         txn = build_fiat_txn(
#     #             victim, receiver, amount, ts,
#     #             transfer_type=tt,
#     #             transfer_network=random.choices(
#     #                 ["domestic", "international"], weights=[0.70, 0.30]
#     #             )[0],
#     #             is_fraud=True,
#     #             fraud_type="account_takeover",
#     #             override_device_hash=override_device,
#     #             override_ip_hash=override_ip,
#     #         )
#     #         txns.append(txn)

#     #     return txns

#     def _inject_ato(self, victim, day) -> list:
#         """
#         CHANGE: 30% of ATO bursts use existing counterparties for some transfers.
#         Real ATOs: attacker sometimes uses the victim's existing payment patterns
#         to stay under detection thresholds (known-payee fraud). The velocity
#         burst remains the primary signal.
    
#         is_first_time_receiver will be True for ~70% of ATO transactions
#         (down from 100%). The velocity signal (3-5 rapid txns) remains strong.
#         """
#         attacker_took_session = random.random() < 0.20
#         override_device = None if attacker_took_session else device_hash(fake.uuid4())
#         override_ip     = None if attacker_took_session else ip_hash(fake.ipv4_private())
    
#         base_ts = rand_ts(day)
#         txns    = []
#         burst_min, burst_max = _FRAUD_PARAMS["burst_count"]
#         n_transfers = random.randint(burst_min, burst_max)
    
#         # 30% of ATO attacks use a mix of new + existing receivers
#         use_existing_mix = random.random() < 0.30
    
#         for i in range(n_transfers):
#             amount = sample_amount(victim["archetype"])
#             ts     = base_ts + timedelta(minutes=i * random.randint(1, 3))
#             tt     = sample_transfer_type()
    
#             # Use existing counterparty for some transfers (known-payee fraud)
#             if use_existing_mix and victim["counterparties"] and random.random() < 0.40:
#                 receiver = random.choice(victim["counterparties"])
#             else:
#                 receiver = identity_hash(fake.email())
    
#             txn = build_fiat_txn(
#                 victim, receiver, amount, ts,
#                 transfer_type=tt,
#                 transfer_network=random.choices(
#                     ["domestic", "international"], weights=[0.70, 0.30]
#                 )[0],
#                 is_fraud=True,
#                 fraud_type="account_takeover",
#                 override_device_hash=override_device,
#                 override_ip_hash=override_ip,
#             )
#             txns.append(txn)
    
#         return txns
#     # ── Fraud Pattern 2: Mule Network ────────────────────────────────────────
#     # Detectable by: graph topology (fan-in on aggregator), not individual txn features

#     # def _inject_mule_network(self, identities_pool, day) -> list:
#     #     """
#     #     PATH B CHANGES:
#     #       - Stage 1 inbound: archetype-normal amounts (not fixed $2K-$10K)
#     #       - Stage 1 transfer type: same distribution as normal
#     #       - Stage 2 mule→aggregator: variable cut (mule keeps 5-20%)
#     #                                  some mules defect (15% skip forwarding)
#     #                                  timing is irregular (1-36 hours)
#     #       - Stage 2 transfer type: same distribution as normal
#     #       - No fraud-hour bias

#     #     What makes this detectable:
#     #       Graph: fraud_source → N mules → 1 aggregator (star + funnel topology)
#     #       fan_in_ratio on aggregator is very high
#     #       receiver_in_degree on aggregator anomalous for its account age
#     #       All stage 1 txns share fraud_network_id
#     #       Aggregator cashout → cross_modality_fraud_id
#     #     """
#     #     network_id = f"mule_{uuid.uuid4().hex[:8]}"
#     #     n_mules    = random.randint(5, 10)
#     #     mules      = random.sample(identities_pool, min(n_mules, len(identities_pool)))

#     #     # Mules are new recruits — low account age (the graph signal)
#     #     for m in mules:
#     #         m["account_age_days"] = random.randint(1, 45)
#     #         m["balance"]          = 0.0

#     #     aggregator = random.choice(mules)
#     #     base_ts    = rand_ts(day)   # any hour — same distribution as legit

#     #     txns             = []
#     #     total_aggregated = 0.0
#     #     fraud_source     = random.choice(identities_pool)

#     #     # Stage 1: Fraud source → each mule
#     #     # Amounts are normal for fraud_source's archetype — not obviously large
#     #     for i, mule in enumerate(mules):
#     #         amount = sample_amount(fraud_source["archetype"])
#     #         ts     = base_ts + timedelta(minutes=i * random.randint(5, 20))
#     #         txn    = build_fiat_txn(
#     #             fraud_source, mule["identity_hash"], amount, ts,
#     #             transfer_type=sample_transfer_type(),
#     #             transfer_network=random.choices(
#     #                 ["same_bank", "domestic"], weights=[0.60, 0.40]
#     #             )[0],
#     #             is_fraud=True,
#     #             fraud_type="mule_network",
#     #             fraud_network_id=network_id,
#     #         )
#     #         mule["balance"] += amount
#     #         total_aggregated += amount
#     #         txns.append(txn)

#     #     # Stage 2: Each mule → aggregator
#     #     # Mule keeps a variable cut (5-20%), some defect entirely
#     #     aggregated_to_agg = 0.0
#     #     for mule in mules:
#     #         if mule["identity_hash"] == aggregator["identity_hash"]:
#     #             continue

#     #         # 15% of mules defect (keep the money, disappear)
#     #         if random.random() < 0.15:
#     #             continue

#     #         # Mule cut: 5-20% of what they received
#     #         mule_cut      = random.uniform(0.05, 0.20)
#     #         forward_amount = mule["balance"] * (1.0 - mule_cut)

#     #         # Irregular timing: anywhere from 1 hour to 36 hours later
#     #         delay_hours = random.uniform(1.0, 36.0)
#     #         ts          = base_ts + timedelta(hours=delay_hours)

#     #         txn = build_fiat_txn(
#     #             mule, aggregator["identity_hash"], forward_amount, ts,
#     #             transfer_type=sample_transfer_type(),
#     #             transfer_network=random.choices(
#     #                 ["same_bank", "domestic"], weights=[0.50, 0.50]
#     #             )[0],
#     #             is_fraud=True,
#     #             fraud_type="mule_network",
#     #             fraud_network_id=network_id,
#     #         )
#     #         aggregated_to_agg += forward_amount
#     #         txns.append(txn)

#     #     # Stage 3: Aggregator cashout (becomes cross-modal bridge)
#     #     if aggregated_to_agg > 0:
#     #         cashout_ts = base_ts + timedelta(hours=random.uniform(12, 48))
#     #         bridge_id  = str(uuid.uuid4())
#     #         cashout    = build_fiat_txn(
#     #             aggregator, identity_hash(fake.email()), aggregated_to_agg, cashout_ts,
#     #             transfer_type=sample_transfer_type(),
#     #             transfer_network=random.choices(
#     #                 ["domestic", "international"], weights=[0.50, 0.50]
#     #             )[0],
#     #             is_fraud=True,
#     #             fraud_type="laundering_fiat_leg",
#     #             fraud_network_id=network_id,
#     #             cross_modality_fraud_id=bridge_id,
#     #         )
#     #         txns.append(cashout)
#     #         self.cross_modal_bridges.append({
#     #             "bridge_id":  bridge_id,
#     #             "user_hash":  aggregator["identity_hash"],
#     #             "amount":     aggregated_to_agg,
#     #             "ts":         cashout_ts,
#     #             "pattern":    "immediate_laundering",
#     #             "network_id": network_id,
#     #         })

#     #     return txns
#     def _inject_mule_network(self, identities_pool, day) -> list:
#         """
#         CHANGE: Give mules pre-existing counterparties so fraud_source
#         is NOT always a first-time receiver for them.
    
#         Real mule recruitment: recruits are briefed to make a few test
#         transactions with the organiser before the main event. They also
#         have existing payment history from their normal life.
    
#         is_first_time_receiver on mule accounts will be False ~35% of the time.
#         The graph topology (fan-in on aggregator) remains the primary signal.
#         """
#         network_id = f"mule_{uuid.uuid4().hex[:8]}"
#         n_mules    = random.randint(5, 10)
#         mules      = random.sample(identities_pool, min(n_mules, len(identities_pool)))
    
#         fraud_source = random.choice(identities_pool)
    
#         for m in mules:
#             m["account_age_days"] = random.randint(1, 45)
#             m["balance"]          = 0.0
#             # 40% of mules have a prior relationship with the fraud source
#             # (test transaction, prior interaction) — not first-time receiver
#             if random.random() < 0.40 and fraud_source["identity_hash"] not in m["counterparties"]:
#                 m["counterparties"].append(fraud_source["identity_hash"])
    
#         aggregator = random.choice(mules)
#         base_ts    = rand_ts(day)
#         txns       = []
#         total_aggregated = 0.0
    
#         # Stage 1: Fraud source → each mule
#         for i, mule in enumerate(mules):
#             amount = sample_amount(fraud_source["archetype"])
#             ts     = base_ts + timedelta(minutes=i * random.randint(5, 20))
#             txn    = build_fiat_txn(
#                 fraud_source, mule["identity_hash"], amount, ts,
#                 transfer_type=sample_transfer_type(),
#                 transfer_network=random.choices(
#                     ["same_bank", "domestic"], weights=[0.60, 0.40]
#                 )[0],
#                 is_fraud=True,
#                 fraud_type="mule_network",
#                 fraud_network_id=network_id,
#             )
#             mule["balance"] += amount
#             total_aggregated += amount
#             txns.append(txn)
    
#         # Stage 2: Each mule → aggregator
#         aggregated_to_agg = 0.0
#         for mule in mules:
#             if mule["identity_hash"] == aggregator["identity_hash"]:
#                 continue
#             if random.random() < 0.15:
#                 continue  # defect
#             mule_cut       = random.uniform(0.05, 0.20)
#             forward_amount = mule["balance"] * (1.0 - mule_cut)
#             delay_hours    = random.uniform(1.0, 36.0)
#             ts             = base_ts + timedelta(hours=delay_hours)
#             txn = build_fiat_txn(
#                 mule, aggregator["identity_hash"], forward_amount, ts,
#                 transfer_type=sample_transfer_type(),
#                 transfer_network=random.choices(
#                     ["same_bank", "domestic"], weights=[0.50, 0.50]
#                 )[0],
#                 is_fraud=True,
#                 fraud_type="mule_network",
#                 fraud_network_id=network_id,
#             )
#             aggregated_to_agg += forward_amount
#             txns.append(txn)
    
#         # Stage 3: Aggregator cashout
#         if aggregated_to_agg > 0:
#             cashout_ts = base_ts + timedelta(hours=random.uniform(12, 48))
#             bridge_id  = str(uuid.uuid4())
#             cashout    = build_fiat_txn(
#                 aggregator, identity_hash(fake.email()), aggregated_to_agg, cashout_ts,
#                 transfer_type=sample_transfer_type(),
#                 transfer_network=random.choices(
#                     ["domestic", "international"], weights=[0.50, 0.50]
#                 )[0],
#                 is_fraud=True,
#                 fraud_type="laundering_fiat_leg",
#                 fraud_network_id=network_id,
#                 cross_modality_fraud_id=bridge_id,
#             )
#             txns.append(cashout)
#             self.cross_modal_bridges.append({
#                 "bridge_id":  bridge_id,
#                 "user_hash":  aggregator["identity_hash"],
#                 "amount":     aggregated_to_agg,
#                 "ts":         cashout_ts,
#                 "pattern":    "immediate_laundering",
#                 "network_id": network_id,
#             })
    
#         return txns


#     # ── Fraud Pattern 3: Synthetic Identity ──────────────────────────────────
#     # Detectable by: account_age < 90 days + velocity spike at drain time

#     def _inject_synthetic_identity(self, day) -> list:
#         """
#         PATH B CHANGES:
#           - Trust-building phase: real small transactions (not skipped)
#           - Inbound transfer: matches the amount range for salary_worker archetype
#           - Drain: partial (60-85%), not 100% — fraudster leaves some to avoid triggers
#           - Drain amount: normal archetype amount scaled by drain factor
#           - Transfer types: normal distribution

#         What makes this detectable:
#           account_age_days: < 90 days (short-lived account)
#           velocity_ratio: sudden spike on drain day vs quiet history
#           is_first_time_receiver: True on drain recipient
#           sender_lifetime_txn_count: low (new account)
#           Graph: low in-degree, zero repeat receivers before drain
#         """
#         synth = {
#             "identity_hash":     identity_hash(fake.email()),
#             "archetype":         "salary_worker",
#             "account_age_days":  random.randint(30, 89),
#             "kyc_status":        "pending",
#             "income_monthly":    3000,
#             "velocity_baseline": 1.0,
#             "balance":           random.uniform(200, 1500),
#             "counterparties":    [identity_hash(fake.email()) for _ in range(2)],
#             "device_hash":       device_hash(fake.uuid4()),
#             "ip_hash":           ip_hash(fake.ipv4()),
#             "_txn_count_30d":    random.randint(3, 12),
#             "_amount_sum_30d":   random.uniform(300, 1200),
#             "_txn_history":      [],
#         }

#         txns = []

#         # Trust-building: 2-4 small legitimate transactions before the drain
#         n_legit = random.randint(2, 4)
#         for i in range(n_legit):
#             legit_day = max(0, day - random.randint(1, 14))
#             legit_amount = sample_amount("student")  # small amounts
#             legit_receiver = random.choice(synth["counterparties"])
#             txn = build_fiat_txn(
#                 synth, legit_receiver, legit_amount, rand_ts(legit_day),
#                 transfer_type=sample_transfer_type(),
#                 is_fraud=False,   # Trust-building txns are NOT labeled fraud
#                 fraud_type=None,
#             )
#             txns.append(txn)

#         # Inbound: large transfer received (we record sender side as synthetic)
#         inbound_amount = sample_amount("small_business")   # realistic large amount
#         inbound_ts     = rand_ts(day)
#         synth["balance"] += inbound_amount

#         # Record the synthetic identity as sender in a small dummy txn
#         # (the actual inbound appears in normal txn flow from the compromised sender)
#         # Drain: partial withdrawal (60-85% of balance)
#         drain_factor = random.uniform(0.60, 0.85)
#         drain_amount = synth["balance"] * drain_factor
#         drain_ts     = inbound_ts + timedelta(hours=random.uniform(2, 12))

#         drain_txn = build_fiat_txn(
#             synth, identity_hash(fake.email()), drain_amount, drain_ts,
#             transfer_type=sample_transfer_type(),
#             transfer_network=random.choices(
#                 ["domestic", "international"], weights=[0.60, 0.40]
#             )[0],
#             is_fraud=True,
#             fraud_type="synthetic_identity",
#         )
#         txns.append(drain_txn)
#         return txns

#     # ── Fraud Pattern 4: BEC ─────────────────────────────────────────────────

#     def _inject_bec(self, business, day) -> list:
#         """
#         PATH B CHANGES:
#           - Amount derived from business archetype (not fixed $10K-$500K)
#           - Transfer type: normal distribution (not always wire)
#           - The signal is the receiver: brand-new account, never seen before

#         What makes this detectable:
#           is_first_time_receiver: True (mule account is new)
#           receiver_in_degree: 0 (nobody has sent to this mule before)
#           amount_ratio: can be high if larger than typical for this business
#         """
#         mule_hash = identity_hash(fake.email())
#         # Scale BEC amount: 1-5x the business's normal transaction amount
#         base_amount = sample_amount("small_business")
#         amount      = base_amount * random.uniform(1.0, 5.0)
#         ts          = rand_ts(day)

#         return [build_fiat_txn(
#             business, mule_hash, amount, ts,
#             transfer_type=sample_transfer_type(),
#             transfer_network=random.choices(
#                 ["domestic", "international"], weights=[0.70, 0.30]
#             )[0],
#             is_fraud=True,
#             fraud_type="bec",
#         )]

#     # ── Structuring ──────────────────────────────────────────────────────────

#     def _inject_structuring(self, user, day) -> list:
#         """
#         PATH B: amounts are archetype-normal (not always $2.5K-$9.5K).
#         Signal: multiple transactions to different receivers in short window.
#         """
#         bridge_id = str(uuid.uuid4())
#         txns      = []
#         n_splits  = random.randint(3, 6)
#         total     = 0.0

#         for i in range(n_splits):
#             amount = sample_amount(user["archetype"]) * random.uniform(0.3, 0.8)
#             ts     = rand_ts(max(0, day - (n_splits - i)))
#             txn    = build_fiat_txn(
#                 user, identity_hash(fake.email()), amount, ts,
#                 transfer_type=sample_transfer_type(),
#                 is_fraud=True,
#                 fraud_type="structuring",
#                 cross_modality_fraud_id=bridge_id,
#             )
#             txns.append(txn)
#             total += amount

#         self.cross_modal_bridges.append({
#             "bridge_id": bridge_id,
#             "user_hash": user["identity_hash"],
#             "amount":    total,
#             "ts":        rand_ts(day),
#             "pattern":   "structuring_to_crypto",
#         })
#         return txns

#     def _inject_reverse_cashout(self, user, day) -> list:
#         bridge_id = str(uuid.uuid4())
#         amount    = sample_amount("small_business") * random.uniform(2, 8)
#         ts        = rand_ts(day)
#         txn       = build_fiat_txn(
#             user, identity_hash(fake.email()), amount, ts,
#             transfer_type=sample_transfer_type(),
#             transfer_network=random.choices(
#                 ["domestic", "international"], weights=[0.50, 0.50]
#             )[0],
#             is_fraud=True,
#             fraud_type="laundering_fiat_leg",
#             cross_modality_fraud_id=bridge_id,
#         )
#         self.cross_modal_bridges.append({
#             "bridge_id": bridge_id,
#             "user_hash": user["identity_hash"],
#             "amount":    amount * 1.02,
#             "ts":        ts - timedelta(hours=random.uniform(6, 48)),
#             "pattern":   "crypto_to_fiat",
#         })
#         return [txn]

#     def run(self) -> list:
#         active     = random.sample(self.identities, int(len(self.identities) * 0.98))
#         businesses = [u for u in active if u["archetype"] == "small_business"]

#         anomaly_users = set(random.sample(range(len(active)), int(len(active) * 0.05)))

#         fraud_budget = {
#             "account_takeover":   int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.40),
#             "mule_network":       int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.30),
#             "synthetic_identity": int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.15),
#             "bec":                int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.15),
#         }
#         fraud_injected       = defaultdict(int)
#         structuring_injected = 0
#         reverse_injected     = 0

#         print(f"\nRunning NeoBank simulator ({SIM_DAYS} days, {len(active):,} users)...")
#         for day in tqdm(range(SIM_DAYS)):
#             for idx, user in enumerate(active):
#                 cfg        = ARCHETYPES[user["archetype"]]
#                 is_weekday = (START_TIME + timedelta(days=day)).weekday() < 5
#                 rate       = cfg["txn_per_day"] if is_weekday else cfg["txn_per_day"] * 0.4
#                 n_txns     = int(rng.poisson(rate))
#                 for _ in range(n_txns):
#                     self.transactions.append(self._normal_txn(user, day))
#                 if idx in anomaly_users and day == 15:
#                     self.transactions.append(self._micro_anomaly(user, day))

#             daily_ato  = fraud_budget["account_takeover"]  // SIM_DAYS
#             daily_mule = fraud_budget["mule_network"]       // SIM_DAYS

#             for _ in range(daily_ato):
#                 if fraud_injected["account_takeover"] < fraud_budget["account_takeover"]:
#                     self.transactions.extend(self._inject_ato(random.choice(active), day))
#                     fraud_injected["account_takeover"] += 1

#             for _ in range(max(1, daily_mule // 8)):
#                 if fraud_injected["mule_network"] < fraud_budget["mule_network"]:
#                     pool = random.sample(active, min(15, len(active)))
#                     self.transactions.extend(self._inject_mule_network(pool, day))
#                     fraud_injected["mule_network"] += 8

#             for _ in range(2):
#                 if fraud_injected["synthetic_identity"] < fraud_budget["synthetic_identity"]:
#                     self.transactions.extend(self._inject_synthetic_identity(day))
#                     fraud_injected["synthetic_identity"] += 2

#             if businesses and fraud_injected["bec"] < fraud_budget["bec"]:
#                 self.transactions.extend(self._inject_bec(random.choice(businesses), day))
#                 fraud_injected["bec"] += 1

#             if structuring_injected < 15:
#                 self.transactions.extend(self._inject_structuring(random.choice(active), day))
#                 structuring_injected += 1

#             if day % 3 == 0 and reverse_injected < 10:
#                 self.transactions.extend(self._inject_reverse_cashout(random.choice(active), day))
#                 reverse_injected += 1

#         return self.transactions


# # ─────────────────────────────────────────────────────────────────────────────
# # CRYPTOEX SIMULATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class CryptoExSimulator:
#     def __init__(self, identities, bridges):
#         self.identities  = identities
#         self.bridges     = bridges
#         self.transactions = []

#     def _normal_txn(self, user, day):
#         arch   = CRYPTO_ARCHETYPES[user.get("crypto_archetype", "hodler")]
#         lo, hi = arch["amount_range"]
#         amount = random.uniform(lo, hi)
#         return build_crypto_txn(
#             user, sha256(user["crypto_wallet"] + "sender"), sha256(fake_wallet()),
#             amount, rand_ts(day),
#             transfer_type=random.choice(["withdrawal", "deposit", "transfer"]),
#             receiver_wallet_type=random.choices(
#                 ["user_wallet", "exchange", "defi_contract"],
#                 weights=[0.60, 0.30, 0.10]
#             )[0],
#         )

#     def _inject_mixer(self, user, day):
#         amount     = random.uniform(10000, 500000)
#         ts         = rand_ts(day)
#         network_id = f"mixer_{uuid.uuid4().hex[:8]}"
#         mixer_addr = sha256(random.choice(MIXER_ADDRESSES))

#         send = build_crypto_txn(
#             user, sha256(user["crypto_wallet"]), mixer_addr, amount, ts,
#             transfer_type="withdrawal", receiver_wallet_type="mixer",
#             known_mixer_interaction=True, receiver_wallet_age_days=9999,
#             is_fraud=True, fraud_type="mixer_usage", fraud_network_id=network_id,
#         )
#         receive = build_crypto_txn(
#             user, mixer_addr, sha256(fake_wallet()),
#             amount * random.uniform(0.95, 1.0),
#             ts + timedelta(hours=random.uniform(2, 48)),
#             transfer_type="deposit", receiver_wallet_type="user_wallet",
#             known_mixer_interaction=True, receiver_wallet_age_days=0,
#             is_fraud=True, fraud_type="mixer_usage", fraud_network_id=network_id,
#         )
#         return [send, receive]

#     def _inject_rug_pull(self, day):
#         network_id = f"rugpull_{uuid.uuid4().hex[:8]}"
#         contract   = sha256(fake_wallet())
#         dev_wallet = sha256(fake_wallet())
#         n_victims  = random.randint(10, 50)
#         pool_total = 0.0
#         txns       = []

#         for _ in range(n_victims):
#             victim_amount = random.uniform(100, 10000)
#             pool_total   += victim_amount
#             fake_user = {
#                 "identity_hash": identity_hash(fake.email()), "crypto_wallet": fake_wallet(),
#                 "crypto_archetype": "hodler", "account_age_days": random.randint(100, 1000),
#                 "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
#                 "_txn_count_30d": random.randint(1, 20), "_amount_sum_30d": random.uniform(500, 10000),
#             }
#             txns.append(build_crypto_txn(
#                 fake_user, sha256(fake_wallet()), contract, victim_amount,
#                 rand_ts(day - random.randint(0, 14)),
#                 receiver_wallet_type="contract", receiver_wallet_age_days=random.randint(1, 30),
#                 is_fraud=True, fraud_type="rug_pull", fraud_network_id=network_id,
#             ))

#         dev_user = {
#             "identity_hash": identity_hash(fake.email()), "crypto_wallet": dev_wallet,
#             "crypto_archetype": "trader", "account_age_days": random.randint(30, 365),
#             "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
#             "_txn_count_30d": 0, "_amount_sum_30d": 0,
#         }
#         txns.append(build_crypto_txn(
#             dev_user, contract, dev_wallet, pool_total, rand_ts(day),
#             transfer_type="withdrawal", receiver_wallet_type="user_wallet",
#             receiver_wallet_age_days=0, is_fraud=True,
#             fraud_type="rug_pull", fraud_network_id=network_id,
#         ))
#         return txns

#     def _inject_phishing(self, user, day):
#         network_id         = f"phish_{uuid.uuid4().hex[:8]}"
#         malicious_contract = sha256(fake_wallet())
#         attacker_wallet    = sha256(fake_wallet())
#         drain_amount       = user.get("balance", random.uniform(1000, 100000))
#         ts                 = rand_ts(day)

#         approval = build_crypto_txn(
#             user, sha256(user["crypto_wallet"]), malicious_contract, 0.0, ts,
#             transfer_type="contract_approval", receiver_wallet_type="contract",
#             receiver_wallet_age_days=random.randint(1, 14),
#             is_fraud=True, fraud_type="phishing", fraud_network_id=network_id,
#         )
#         drain = build_crypto_txn(
#             user, malicious_contract, attacker_wallet, drain_amount,
#             ts + timedelta(seconds=random.randint(10, 120)),
#             transfer_type="withdrawal", receiver_wallet_type="user_wallet",
#             receiver_wallet_age_days=0, is_fraud=True,
#             fraud_type="phishing", fraud_network_id=network_id,
#         )
#         drain["immediate_withdrawal"] = True
#         return [approval, drain]

#     def _inject_pig_butchering(self, day):
#         network_id     = f"pig_{uuid.uuid4().hex[:8]}"
#         scammer_wallet = sha256(fake_wallet())
#         n_victims      = random.randint(10, 40)
#         txns           = []

#         scammer_user = {
#             "identity_hash": identity_hash(fake.email()), "crypto_wallet": scammer_wallet,
#             "crypto_archetype": "trader", "account_age_days": random.randint(30, 180),
#             "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
#             "_txn_count_30d": 0, "_amount_sum_30d": 0,
#         }
#         for _ in range(n_victims):
#             victim_user = {
#                 "identity_hash": identity_hash(fake.email()), "crypto_wallet": fake_wallet(),
#                 "crypto_archetype": "hodler", "account_age_days": random.randint(100, 2000),
#                 "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
#                 "_txn_count_30d": 0, "_amount_sum_30d": 0,
#             }
#             base_amount = random.uniform(500, 2000)
#             for p in range(random.randint(3, 8)):
#                 amount = base_amount * (1.3 ** p)
#                 txns.append(build_crypto_txn(
#                     victim_user, sha256(victim_user["crypto_wallet"]), scammer_wallet,
#                     amount, rand_ts(max(0, day - random.randint(0, 30))),
#                     receiver_wallet_type="user_wallet",
#                     is_fraud=True, fraud_type="pig_butchering", fraud_network_id=network_id,
#                 ))

#         txns.extend(self._inject_mixer(scammer_user, day))
#         return txns

#     def _process_bridges(self):
#         for bridge in self.bridges:
#             user_hash = bridge["user_hash"]
#             amount    = bridge["amount"]
#             fiat_ts   = bridge["ts"]
#             pattern   = bridge["pattern"]

#             user = next((u for u in self.identities if u["identity_hash"] == user_hash), None)
#             if user is None:
#                 user = {
#                     "identity_hash": user_hash, "crypto_wallet": fake_wallet(),
#                     "crypto_archetype": "trader", "account_age_days": random.randint(1, 30),
#                     "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
#                     "_txn_count_30d": 0, "_amount_sum_30d": 0,
#                 }

#             if pattern == "immediate_laundering":
#                 crypto_ts   = fiat_ts + timedelta(minutes=random.randint(10, 60))
#                 delay_label = "immediate_laundering"
#             elif pattern == "delayed_laundering":
#                 crypto_ts   = fiat_ts + timedelta(hours=random.uniform(24, 72))
#                 delay_label = "delayed_laundering"
#             elif pattern == "structuring_to_crypto":
#                 crypto_ts   = fiat_ts + timedelta(hours=random.uniform(1, 12))
#                 delay_label = "structuring_to_crypto"
#             else:
#                 crypto_ts   = fiat_ts
#                 delay_label = "crypto_to_fiat"

#             crypto_txn = build_crypto_txn(
#                 user, sha256(user["crypto_wallet"]), sha256(fake_wallet()),
#                 amount * random.uniform(0.95, 0.99), crypto_ts,
#                 transfer_type="withdrawal", receiver_wallet_type="user_wallet",
#                 receiver_wallet_age_days=0,
#                 is_fraud=True, fraud_type="laundering_crypto_leg",
#                 fraud_network_id=bridge.get("network_id"),
#                 cross_modality_fraud_id=bridge["bridge_id"],
#             )
#             crypto_txn["cross_modality_pattern"] = delay_label
#             crypto_txn["immediate_withdrawal"]   = True
#             self.transactions.append(crypto_txn)

#     def run(self) -> list:
#         active = [u for u in self.identities if u.get("has_crypto", False)]
#         if not active:
#             active = random.sample(self.identities, int(len(self.identities) * 0.35))

#         print(f"\nRunning CryptoEx simulator ({len(active):,} users, {SIM_DAYS} days)...")
#         estimated     = len(active) * SIM_DAYS * 0.5
#         fraud_budget  = int(estimated * CRYPTO_FRAUD_RATE / (1 - CRYPTO_FRAUD_RATE))
#         mixer_budget  = int(fraud_budget * 0.35)
#         rug_budget    = int(fraud_budget * 0.25)
#         phish_budget  = int(fraud_budget * 0.20)
#         pig_budget    = int(fraud_budget * 0.20)

#         mixer_count = rug_count = phish_count = pig_count = 0

#         for day in tqdm(range(SIM_DAYS)):
#             for user in active:
#                 arch = CRYPTO_ARCHETYPES[user.get("crypto_archetype", "hodler")]
#                 if random.random() < arch["txn_per_day"]:
#                     self.transactions.append(self._normal_txn(user, day))

#             if mixer_count < mixer_budget:
#                 self.transactions.extend(self._inject_mixer(random.choice(active), day))
#                 mixer_count += 2

#             if rug_count < rug_budget and day % 3 == 0:
#                 self.transactions.extend(self._inject_rug_pull(day))
#                 rug_count += 30

#             if phish_count < phish_budget:
#                 self.transactions.extend(self._inject_phishing(random.choice(active), day))
#                 phish_count += 2

#             if pig_count < pig_budget and day % 5 == 0:
#                 self.transactions.extend(self._inject_pig_butchering(day))
#                 pig_count += 50

#         print("  → Processing cross-modal bridges...")
#         self._process_bridges()
#         return self.transactions


# # ─────────────────────────────────────────────────────────────────────────────
# # FINBRIDGE SIMULATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class FinBridgeSimulator:
#     def __init__(self, identities):
#         self.identities  = identities
#         self.transactions = []

#     def run(self) -> list:
#         active = random.sample(self.identities, int(len(self.identities) * 0.15))
#         print(f"\nRunning FinBridge simulator ({len(active):,} users)...")

#         for user in tqdm(active):
#             for _ in range(random.randint(1, 5)):
#                 amount    = sample_amount(user["archetype"])
#                 day       = random.randint(0, SIM_DAYS - 1)
#                 ts        = rand_ts(day)
#                 tx_link   = uuid.uuid4().hex[:8]
#                 is_fraud  = random.random() < (FIAT_FRAUD_RATE * 1.5)
#                 bridge_id = str(uuid.uuid4()) if is_fraud else None

#                 fiat_rec = build_fiat_txn(
#                     user, sha256("finbridge_internal"), amount, ts,
#                     transfer_type=sample_transfer_type(),
#                     transfer_network="domestic",
#                     is_fraud=is_fraud,
#                     fraud_type="laundering_fiat_leg" if is_fraud else None,
#                     cross_modality_fraud_id=bridge_id,
#                 )
#                 fiat_rec["transaction_id"] = f"finbridge_f_{tx_link}"
#                 fiat_rec["client_id"]      = "finbridge_prod"
#                 self.transactions.append(fiat_rec)

#                 crypto_rec = build_crypto_txn(
#                     user, sha256(user["crypto_wallet"]), sha256(fake_wallet()),
#                     amount * 0.99, ts + timedelta(seconds=30),
#                     transfer_type="crypto_purchase", receiver_wallet_type="user_wallet",
#                     is_fraud=is_fraud,
#                     fraud_type="laundering_crypto_leg" if is_fraud else None,
#                     cross_modality_fraud_id=bridge_id,
#                 )
#                 crypto_rec["transaction_id"]         = f"finbridge_c_{tx_link}"
#                 crypto_rec["client_id"]               = "finbridge_prod"
#                 crypto_rec["linked_fiat_transaction"] = fiat_rec["transaction_id"]
#                 self.transactions.append(crypto_rec)

#         return self.transactions


# # ─────────────────────────────────────────────────────────────────────────────
# # VALIDATION
# # ─────────────────────────────────────────────────────────────────────────────

# def validate_and_summarize(df):
#     print("\n" + "="*60)
#     print("SIMULATION SUMMARY")
#     print("="*60)
#     total = len(df)
#     fraud = df["is_fraud"].sum()
#     print(f"Total: {total:,}  |  Fraud: {fraud:,} ({fraud/total*100:.2f}%)")

#     print("\n── By client ──")
#     for client, grp in df.groupby("client_id"):
#         f = grp["is_fraud"].sum()
#         print(f"  {client:<20} {len(grp):>8,}  fraud: {f:>6,} ({f/len(grp)*100:.1f}%)")

#     print("\n── Fraud types ──")
#     for ft, cnt in df[df["is_fraud"]]["fraud_type"].value_counts().items():
#         print(f"  {str(ft):<35} {cnt:>6,}")

#     print("\n── Cohen's d — top 10 separating features ──")
#     print("  (d > 1.0 = simulator artifact, d < 0.5 = realistic overlap)")
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     numeric_cols = [c for c in numeric_cols if c not in ["is_fraud"]]
#     results = []
#     fraud_df = df[df["is_fraud"] == True]
#     legit_df = df[df["is_fraud"] == False]
#     for col in numeric_cols:
#         fd, ld    = fraud_df[col].dropna(), legit_df[col].dropna()
#         if len(fd) < 10 or len(ld) < 10:
#             continue
#         pooled_std = np.sqrt((fd.std()**2 + ld.std()**2) / 2 + 1e-9)
#         cohens_d   = abs(fd.mean() - ld.mean()) / pooled_std
#         results.append((col, round(cohens_d, 3)))
#     results.sort(key=lambda x: x[1], reverse=True)
#     for col, d in results[:10]:
#         flag = " ← ARTIFACT" if d > 1.0 else (" ← moderate" if d > 0.5 else "")
#         print(f"  {col:<35} d={d:.3f}{flag}")

#     cross = df[df["cross_modality_fraud_id"].notna()]
#     print(f"\n── Cross-modal: {len(cross):,} txns, {cross['cross_modality_fraud_id'].nunique():,} sequences ──")
#     print("="*60)


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("="*60)
#     print("SENTINEL SIMULATOR v3.0 — Path B")
#     print("Fraud defined by sequence + graph topology, not feature flags")
#     print("="*60)

#     identities = build_identity_pool(N_IDENTITIES)

#     neobank   = NeoBankSimulator(identities)
#     fiat_txns = neobank.run()
#     print(f"  NeoBank: {len(fiat_txns):,} txns, {len(neobank.cross_modal_bridges):,} bridges")

#     finbridge   = FinBridgeSimulator(identities)
#     bridge_txns = finbridge.run()
#     print(f"  FinBridge: {len(bridge_txns):,} txns")

#     cryptoex    = CryptoExSimulator(identities, neobank.cross_modal_bridges)
#     crypto_txns = cryptoex.run()
#     print(f"  CryptoEx: {len(crypto_txns):,} txns")

#     all_txns = fiat_txns + bridge_txns + crypto_txns
#     df = pd.DataFrame(all_txns)
#     df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
#     df = df.sort_values("timestamp").reset_index(drop=True)

#     validate_and_summarize(df)

#     out_path = "sentinel_training_data.jsonl"
#     df.to_json(out_path, orient="records", lines=True, date_format="iso")
#     print(f"\nSaved → {out_path}")

#     df[df["is_fraud"]].head(1000).to_json(
#         "sentinel_fraud_sample.jsonl", orient="records", lines=True
#     )
#     print("Saved → sentinel_fraud_sample.jsonl")

"""
Sentinel Training Data Simulator — v3.1 (Path B, fully patched)
================================================================
All changes from v2 and the targeted fixes for:
  - is_first_time_receiver dominance (ATO known-payee mix, mule pre-existing)
  - balance_drain_ratio artifact (3% high-drain legitimate transactions)
  - amount_ratio artifact (lower effective repeat rate = richer user history)
"""

import hashlib
import uuid
import random
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

fake = Faker()
rng  = np.random.default_rng(42)

SHARED_SALT   = os.getenv("SENTINEL_SALT", "sentinel_consortium_2026_v1")
DEVICE_SALT   = os.getenv("DEVICE_SALT",   "sentinel_device_2026_v1")
IP_SALT       = os.getenv("IP_SALT",       "sentinel_ip_2026_v1")

START_TIME        = datetime(2026, 1, 1, 0, 0, 0)
SIM_DAYS          = 30
N_IDENTITIES      = 10_000
TARGET_FIAT_TXN   = 1_500_000
TARGET_CRYPTO_TXN = 500_000
FIAT_FRAUD_RATE   = 0.020
CRYPTO_FRAUD_RATE = 0.030
CROSS_MODAL_RATE  = 0.20
CALIBRATION_PATH  = os.getenv("CALIBRATION_PATH", "data/calibration/calibration.yaml")


def load_calibration(path):
    p = Path(path)
    if not p.exists():
        print(f"[calibration] WARNING: {path} not found — using defaults.")
        return {}
    with open(p) as f:
        cal = yaml.safe_load(f)
    print(f"[calibration] Loaded from {path}")
    return cal or {}

def _build_archetype_lognorm(cal):
    fit      = cal.get("paysim_fit", {})
    base_mu  = fit.get("mu",   12.97)
    base_sg  = fit.get("sigma", 1.30)
    arch_cfg = cal.get("archetype_amount_lognorm", {})
    defaults = {
        "salary_worker":  {"mu_offset": -4.7, "sigma_scale": 0.70},
        "freelancer":     {"mu_offset": -4.2, "sigma_scale": 1.00},
        "small_business": {"mu_offset": -3.2, "sigma_scale": 1.20},
        "retiree":        {"mu_offset": -5.5, "sigma_scale": 0.50},
        "student":        {"mu_offset": -6.0, "sigma_scale": 0.60},
    }
    result = {}
    for arch, fallback in defaults.items():
        cfg   = arch_cfg.get(arch, fallback)
        mu    = base_mu + cfg.get("mu_offset",   fallback["mu_offset"])
        sigma = base_sg * cfg.get("sigma_scale", fallback["sigma_scale"])
        result[arch] = (round(mu, 4), round(sigma, 4))
    return result

def _build_fraud_params(cal):
    raw_burst = cal.get("fraud_burst_count", [3, 5])
    return {"burst_count": (max(3, raw_burst[0]), max(5, raw_burst[1]))}

print("[calibration] Loading...")
_CAL               = load_calibration(CALIBRATION_PATH)
_ARCHETYPE_LOGNORM = _build_archetype_lognorm(_CAL)
_FRAUD_PARAMS      = _build_fraud_params(_CAL)
print("[calibration] Ready.\n")

ARCHETYPES = {
    "salary_worker":  {"weight": 0.40, "txn_per_day": 2.5,  "income_range": (3000,  8000),  "counterparties": (5,  15), "repeat_rate": 0.80, "kyc_status": "verified"},
    "freelancer":     {"weight": 0.25, "txn_per_day": 5.0,  "income_range": (2000,  15000), "counterparties": (10, 30), "repeat_rate": 0.60, "kyc_status": "verified"},
    "small_business": {"weight": 0.15, "txn_per_day": 12.0, "income_range": (10000, 100000),"counterparties": (20, 50), "repeat_rate": 0.50, "kyc_status": "verified"},
    "retiree":        {"weight": 0.10, "txn_per_day": 1.0,  "income_range": (1500,  4000),  "counterparties": (3,  8),  "repeat_rate": 0.90, "kyc_status": "verified"},
    "student":        {"weight": 0.10, "txn_per_day": 1.5,  "income_range": (500,   2000),  "counterparties": (2,  6),  "repeat_rate": 0.75, "kyc_status": "pending"},
}
for _arch, _lognorm in _ARCHETYPE_LOGNORM.items():
    if _arch in ARCHETYPES:
        ARCHETYPES[_arch]["amount_lognorm"] = _lognorm

CRYPTO_ARCHETYPES = {
    "hodler":   {"weight": 0.30, "txn_per_day": 0.05, "amount_range": (500,   5000)},
    "trader":   {"weight": 0.25, "txn_per_day": 1.5,  "amount_range": (1000,  20000)},
    "defi":     {"weight": 0.20, "txn_per_day": 0.5,  "amount_range": (500,   10000)},
    "merchant": {"weight": 0.15, "txn_per_day": 3.0,  "amount_range": (50,    5000)},
    "p2p":      {"weight": 0.10, "txn_per_day": 0.4,  "amount_range": (20,    500)},
}

HOURLY_WEIGHTS = [
    0.05, 0.03, 0.03, 0.03, 0.04, 0.06,
    0.10, 0.20, 0.40, 0.60, 0.80, 0.90,
    0.85, 0.80, 0.90, 1.00, 0.90, 0.75,
    0.55, 0.40, 0.30, 0.20, 0.12, 0.07,
]
_hw_total      = sum(HOURLY_WEIGHTS)
HOURLY_WEIGHTS = [w / _hw_total for w in HOURLY_WEIGHTS]

MIXER_ADDRESSES = [
    "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",
    "0x8d12A197cB00D4747a1fe03395095ce2A5CC6819",
    "0xA160cdAB225685dA1d56aa342Ad8841c3b53f291",
]


def sha256(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()

def identity_hash(email: str) -> str:
    return sha256(f"{email}{SHARED_SALT}")

def device_hash(d: str) -> str:
    return sha256(f"{d}{DEVICE_SALT}")

def ip_hash(ip: str) -> str:
    return sha256(f"{ip}{IP_SALT}")

def sample_hour() -> int:
    return int(rng.choice(24, p=HOURLY_WEIGHTS))

def rand_ts(day: int) -> datetime:
    return START_TIME + timedelta(
        days=day, hours=sample_hour(),
        minutes=random.randint(0, 59), seconds=random.randint(0, 59),
    )

def sample_amount(archetype: str) -> float:
    mu, sigma = ARCHETYPES[archetype]["amount_lognorm"]
    return max(10.0, float(rng.lognormal(mu, sigma)))

def sample_transfer_type() -> str:
    return random.choices(["ach", "wire", "p2p", "rtp"], weights=[0.60, 0.20, 0.15, 0.05])[0]

def txn_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def fake_wallet() -> str:
    return f"0x{uuid.uuid4().hex[:40]}"

def fake_tx_hash() -> str:
    return f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:24]}"


def build_identity_pool(n: int) -> list:
    """
    Identity pool with financial obligation profile.
    has_large_recurring, recurring_drain_pct, recurring_receiver drive
    high-drain legitimate transactions that overlap with fraud drain patterns.
    """
    arch_keys    = list(ARCHETYPES.keys())
    arch_weights = [ARCHETYPES[a]["weight"] for a in arch_keys]
    arch_rent_prob = {
        "salary_worker": 0.65, "freelancer": 0.70,
        "small_business": 0.80, "retiree": 0.40, "student": 0.55,
    }
    identities = []
    print(f"Building {n:,} consortium identities...")
    for _ in tqdm(range(n)):
        email = fake.email()
        arch  = random.choices(arch_keys, weights=arch_weights)[0]
        cfg   = ARCHETYPES[arch]
        identities.append({
            "identity_hash":       identity_hash(email),
            "email":               email,
            "archetype":           arch,
            "income_monthly":      random.randint(*cfg["income_range"]),
            "account_age_days":    random.randint(30, 3650),
            "kyc_status":          cfg["kyc_status"],
            "velocity_baseline":   cfg["txn_per_day"] * random.uniform(0.7, 1.3),
            "balance":             random.randint(*cfg["income_range"]) * random.uniform(1.0, 6.0),
            "counterparties":      [identity_hash(fake.email()) for _ in range(random.randint(*cfg["counterparties"]))],
            "device_hash":         device_hash(fake.uuid4()),
            "ip_hash":             ip_hash(fake.ipv4()),
            "has_crypto":          random.random() < 0.35,
            "crypto_wallet":       fake_wallet(),
            "crypto_archetype":    random.choices(
                list(CRYPTO_ARCHETYPES.keys()),
                weights=[v["weight"] for v in CRYPTO_ARCHETYPES.values()]
            )[0],
            # Recurring obligation fields — drive high-drain legit transactions
            "has_large_recurring": random.random() < arch_rent_prob.get(arch, 0.50),
            "recurring_drain_pct": random.uniform(0.35, 0.65),
            "recurring_receiver":  identity_hash(fake.email()),
            "_txn_count_30d":      0,
            "_amount_sum_30d":     0.0,
            "_txn_history":        [],
        })
    return identities


def compute_velocity_features(user: dict, current_ts: datetime, windows: list) -> dict:
    feats   = {}
    history = user["_txn_history"]
    for h in windows:
        cutoff = current_ts - timedelta(hours=h)
        w      = [(ts, amt) for ts, amt in history if ts >= cutoff]
        feats[f"count_{h}h"]      = len(w)
        feats[f"sum_amount_{h}h"] = round(sum(a for _, a in w), 2)
    return feats

def update_user_history(user: dict, ts: datetime, amount: float):
    user["_txn_history"].append((ts, amount))
    cutoff = ts - timedelta(days=7)
    user["_txn_history"] = [(t, a) for t, a in user["_txn_history"] if t >= cutoff]
    user["_txn_count_30d"]  += 1
    user["_amount_sum_30d"] += amount

def velocity_ratio(user: dict, count_24h: int) -> float:
    baseline = user["velocity_baseline"]
    return round(count_24h / baseline, 3) if baseline > 0 else 0.0

def amount_ratio(user: dict, amount: float) -> float:
    if user["_txn_count_30d"] == 0:
        return 1.0
    avg = user["_amount_sum_30d"] / user["_txn_count_30d"]
    return round(amount / avg, 3) if avg > 0 else 1.0


def build_fiat_txn(
    sender, receiver_hash, amount, ts,
    transfer_type="ach", transfer_network="domestic",
    is_fraud=False, fraud_type=None,
    fraud_network_id=None, cross_modality_fraud_id=None,
    override_device_hash=None, override_ip_hash=None,
) -> dict:
    balance_before = max(0.0, sender["balance"])
    balance_after  = max(0.0, balance_before - amount)
    drain_ratio    = round(amount / balance_before, 4) if balance_before > 0 else 0.0
    vel            = compute_velocity_features(sender, ts, [1, 6, 24, 168])
    sender["balance"] = balance_after
    is_first_time = receiver_hash not in sender["counterparties"]
    if not is_first_time:
        sender["counterparties"].append(receiver_hash)
    update_user_history(sender, ts, amount)
    return {
        "transaction_id":            txn_id("neobank"),
        "client_id":                 "neobank_prod",
        "timestamp":                 ts.isoformat() + "Z",
        "modality":                  "fiat",
        "identity_hash":             sender["identity_hash"],
        "sender_hash":               sender["identity_hash"],
        "receiver_hash":             receiver_hash,
        "sender_device_hash":        override_device_hash or sender["device_hash"],
        "sender_ip_hash":            override_ip_hash     or sender["ip_hash"],
        "amount_usd":                round(amount, 2),
        "transfer_type":             transfer_type,
        "transfer_network":          transfer_network,
        "sender_account_age_days":   sender["account_age_days"],
        "sender_kyc_status":         sender["kyc_status"],
        "sender_archetype":          sender["archetype"],
        "sender_lifetime_txn_count": sender["_txn_count_30d"],
        "sender_lifetime_volume_usd": round(sender["_amount_sum_30d"], 2),
        "is_first_time_receiver":    is_first_time,
        "sender_balance_before":     round(balance_before, 2),
        "sender_balance_after":      round(balance_after, 2),
        "balance_drain_ratio":       drain_ratio,
        "count_1h":                  vel["count_1h"],
        "count_6h":                  vel["count_6h"],
        "count_24h":                 vel["count_24h"],
        "count_7d":                  vel["count_168h"],
        "sum_amount_1h":             vel["sum_amount_1h"],
        "sum_amount_24h":            vel["sum_amount_24h"],
        "sum_amount_7d":             vel["sum_amount_168h"],
        "velocity_ratio":            velocity_ratio(sender, vel["count_24h"]),
        "amount_ratio":              amount_ratio(sender, amount),
        "velocity_baseline_daily":   round(sender["velocity_baseline"], 3),
        "hour_of_day":               ts.hour,
        "day_of_week":               ts.weekday(),
        "is_weekend":                ts.weekday() >= 5,
        "is_unusual_hour":           ts.hour < 6 or ts.hour >= 23,
        "is_fraud":                  is_fraud,
        "fraud_type":                fraud_type,
        "fraud_network_id":          fraud_network_id,
        "cross_modality_fraud_id":   cross_modality_fraud_id,
    }


def build_crypto_txn(
    user, sender_wallet_hash, receiver_wallet_hash, amount_usd, ts,
    transfer_type="withdrawal", receiver_wallet_type="user_wallet",
    known_mixer_interaction=False, receiver_wallet_age_days=None,
    is_fraud=False, fraud_type=None,
    fraud_network_id=None, cross_modality_fraud_id=None,
    override_device_hash=None, override_ip_hash=None,
) -> dict:
    cryptocurrency = random.choices(
        ["ETH", "BTC", "USDT", "USDC", "SOL"],
        weights=[0.40, 0.25, 0.20, 0.10, 0.05]
    )[0]
    return {
        "transaction_id":            txn_id("cryptoex"),
        "client_id":                 "cryptoex_prod",
        "timestamp":                 ts.isoformat() + "Z",
        "modality":                  "crypto",
        "identity_hash":             user["identity_hash"],
        "sender_wallet_hash":        sender_wallet_hash,
        "receiver_wallet_hash":      receiver_wallet_hash,
        "user_email_hash":           user["identity_hash"],
        "sender_device_hash":        override_device_hash or user["device_hash"],
        "sender_ip_hash":            override_ip_hash     or user["ip_hash"],
        "amount_usd":                round(amount_usd, 2),
        "amount_crypto":             round(amount_usd / 2800.0, 6),
        "cryptocurrency":            cryptocurrency,
        "blockchain":                "ethereum",
        "transaction_hash":          fake_tx_hash(),
        "gas_price_gwei":            random.randint(10, 150),
        "sender_wallet_age_days":    user.get("account_age_days", 30),
        "sender_lifetime_txn_count": user.get("_txn_count_30d", 0),
        "sender_lifetime_volume_usd": round(user.get("_amount_sum_30d", 0), 2),
        "receiver_wallet_type":      receiver_wallet_type,
        "receiver_wallet_age_days":  receiver_wallet_age_days if receiver_wallet_age_days is not None else random.randint(30, 1000),
        "known_mixer_interaction":   known_mixer_interaction,
        "is_first_time_receiver":    random.random() < 0.3,
        "immediate_withdrawal":      False,
        "hour_of_day":               ts.hour,
        "day_of_week":               ts.weekday(),
        "is_weekend":                ts.weekday() >= 5,
        "is_unusual_hour":           ts.hour < 6 or ts.hour >= 23,
        "is_fraud":                  is_fraud,
        "fraud_type":                fraud_type,
        "fraud_network_id":          fraud_network_id,
        "cross_modality_fraud_id":   cross_modality_fraud_id,
    }


class NeoBankSimulator:
    def __init__(self, identities):
        self.identities          = identities
        self.transactions        = []
        self.cross_modal_bridges = []

    def _normal_txn(self, user, day):
        """
        FIX: 3% of legitimate transactions are high-drain payments.
        This creates overlap between fraud and legit on balance_drain_ratio.
        Also uses repeat_rate * 0.77 to increase new-receiver rate from 20% → 38%,
        reducing is_first_time_receiver's power as a single-feature separator.

        Distribution:
          1%: account-emptying (80-99% drain) → moving banks, emergency, large purchase
          2%: large recurring payment (35-65% drain) → rent, mortgage, tax
         97%: normal small transaction (1-15% drain typical)
        """
        r = random.random()

        if r < 0.01:
            # Account-emptying: 80-99% of balance to a new payee
            # Real: moving banks, paying off a loan, medical emergency
            amount   = user["balance"] * random.uniform(0.80, 0.99)
            amount   = max(10.0, amount)
            receiver = identity_hash(fake.email())  # new bank/payee
            return build_fiat_txn(
                user, receiver, amount, rand_ts(day),
                transfer_type=sample_transfer_type(),
                transfer_network=random.choices(
                    ["domestic", "international"], weights=[0.70, 0.30]
                )[0],
            )

        elif r < 0.03 and user.get("has_large_recurring", False):
            # Large recurring payment: rent, mortgage, tax — 35-65% of balance
            # Always goes to the SAME stable counterparty (landlord, bank, FIRS)
            drain_pct = user.get("recurring_drain_pct", 0.45) * random.uniform(0.85, 1.15)
            drain_pct = min(drain_pct, 0.95)
            amount    = max(10.0, user["balance"] * drain_pct)
            receiver  = user.get("recurring_receiver", identity_hash(fake.email()))
            # Ensure receiver is a known counterparty (stable recurring relationship)
            if receiver not in user["counterparties"]:
                user["counterparties"].append(receiver)
            return build_fiat_txn(
                user, receiver, amount, rand_ts(day),
                transfer_type=random.choices(["ach", "wire"], weights=[0.70, 0.30])[0],
                transfer_network="domestic",
            )

        else:
            # Normal transaction (97% of the time)
            amount = sample_amount(user["archetype"])
            # FIX: reduce effective repeat rate by 23% → more new-receiver legitimate txns
            # salary_worker: 0.80 * 0.77 = 0.62 → 38% new receivers (was 20%)
            # freelancer:    0.60 * 0.77 = 0.46 → 54% new receivers
            # small_business:0.50 * 0.77 = 0.39 → 61% new receivers
            effective_repeat = ARCHETYPES[user["archetype"]]["repeat_rate"] * 0.77
            repeat   = random.random() < effective_repeat
            receiver = random.choice(user["counterparties"]) \
                       if repeat and user["counterparties"] \
                       else identity_hash(fake.email())
            return build_fiat_txn(
                user, receiver, amount, rand_ts(day),
                transfer_type=sample_transfer_type(),
                transfer_network=random.choices(
                    ["same_bank", "domestic", "international"],
                    weights=[0.40, 0.50, 0.10]
                )[0],
            )

    def _micro_anomaly(self, user, day):
        kind = random.choices(
            ["travel_wire", "large_one_time", "income_spike"],
            weights=[0.50, 0.30, 0.20]
        )[0]
        if kind == "travel_wire":
            amount, tt, tn = random.uniform(1000, 8000), "wire", "international"
        elif kind == "large_one_time":
            amount, tt, tn = random.uniform(10000, 50000), "wire", "domestic"
        else:
            amount, tt, tn = user["income_monthly"] * random.uniform(1.5, 3.0), "ach", "domestic"
        return build_fiat_txn(user, identity_hash(fake.email()), amount, rand_ts(day),
                              transfer_type=tt, transfer_network=tn)

    def _inject_ato(self, victim, day) -> list:
        """
        FIX: 30% of ATO bursts use existing counterparties for some transfers
        (known-payee fraud). Reduces is_first_time_receiver from 100% to ~70%.
        Velocity burst (3-5 rapid txns) remains the primary detection signal.
        """
        attacker_took_session = random.random() < 0.20
        override_device = None if attacker_took_session else device_hash(fake.uuid4())
        override_ip     = None if attacker_took_session else ip_hash(fake.ipv4_private())
        base_ts = rand_ts(day)
        txns    = []
        burst_min, burst_max = _FRAUD_PARAMS["burst_count"]
        n_transfers = random.randint(burst_min, burst_max)
        use_existing_mix = random.random() < 0.30

        for i in range(n_transfers):
            amount = sample_amount(victim["archetype"])
            ts     = base_ts + timedelta(minutes=i * random.randint(1, 3))
            if use_existing_mix and victim["counterparties"] and random.random() < 0.40:
                receiver = random.choice(victim["counterparties"])
            else:
                receiver = identity_hash(fake.email())
            txns.append(build_fiat_txn(
                victim, receiver, amount, ts,
                transfer_type=sample_transfer_type(),
                transfer_network=random.choices(
                    ["domestic", "international"], weights=[0.70, 0.30]
                )[0],
                is_fraud=True, fraud_type="account_takeover",
                override_device_hash=override_device,
                override_ip_hash=override_ip,
            ))
        return txns

    def _inject_mule_network(self, identities_pool, day) -> list:
        """
        FIX: 40% of mules have a pre-existing relationship with fraud_source
        (test payment before main event). is_first_time_receiver on mule
        accounts will be False ~35% of the time.
        Graph topology (fan-in on aggregator) remains the primary signal.
        """
        network_id   = f"mule_{uuid.uuid4().hex[:8]}"
        n_mules      = random.randint(5, 10)
        mules        = random.sample(identities_pool, min(n_mules, len(identities_pool)))
        fraud_source = random.choice(identities_pool)

        for m in mules:
            m["account_age_days"] = random.randint(1, 45)
            m["balance"]          = 0.0
            if random.random() < 0.40 and fraud_source["identity_hash"] not in m["counterparties"]:
                m["counterparties"].append(fraud_source["identity_hash"])

        aggregator = random.choice(mules)
        base_ts    = rand_ts(day)
        txns       = []

        for i, mule in enumerate(mules):
            amount = sample_amount(fraud_source["archetype"])
            ts     = base_ts + timedelta(minutes=i * random.randint(5, 20))
            txns.append(build_fiat_txn(
                fraud_source, mule["identity_hash"], amount, ts,
                transfer_type=sample_transfer_type(),
                # transfer_network=random.choices(["same_bank", "domestic"], weights=[0.60, 0.40])[0],
                transfer_network=random.choices(["same_bank", "domestic", "international"], weights=[0.40, 0.50, 0.10])[0],
                is_fraud=True, fraud_type="mule_network", fraud_network_id=network_id,
            ))
            mule["balance"] += amount

        aggregated_to_agg = 0.0
        for mule in mules:
            if mule["identity_hash"] == aggregator["identity_hash"]:
                continue
            if random.random() < 0.15:
                continue
            forward_amount = mule["balance"] * (1.0 - random.uniform(0.05, 0.20))
            ts = base_ts + timedelta(hours=random.uniform(1.0, 36.0))
            txns.append(build_fiat_txn(
                mule, aggregator["identity_hash"], forward_amount, ts,
                transfer_type=sample_transfer_type(),
                # transfer_network=random.choices(["same_bank", "domestic"], weights=[0.50, 0.50])[0],
                transfer_network=random.choices(["same_bank", "domestic", "international"], weights=[0.40, 0.50, 0.10])[0],
                is_fraud=True, fraud_type="mule_network", fraud_network_id=network_id,
            ))
            aggregated_to_agg += forward_amount

        if aggregated_to_agg > 0:
            cashout_ts = base_ts + timedelta(hours=random.uniform(12, 48))
            bridge_id  = str(uuid.uuid4())
            txns.append(build_fiat_txn(
                aggregator, identity_hash(fake.email()), aggregated_to_agg, cashout_ts,
                transfer_type=sample_transfer_type(),
                # transfer_network=random.choices(["domestic", "international"], weights=[0.50, 0.50])[0],
                transfer_network=random.choices(["same_bank", "domestic", "international"], weights=[0.40, 0.50, 0.10])[0],
                is_fraud=True, fraud_type="laundering_fiat_leg",
                fraud_network_id=network_id, cross_modality_fraud_id=bridge_id,
            ))
            self.cross_modal_bridges.append({
                "bridge_id": bridge_id, "user_hash": aggregator["identity_hash"],
                "amount": aggregated_to_agg, "ts": cashout_ts,
                "pattern": "immediate_laundering", "network_id": network_id,
            })
        return txns

    def _inject_synthetic_identity(self, day) -> list:
        synth = {
            "identity_hash":     identity_hash(fake.email()),
            "archetype":         "salary_worker",
            "account_age_days":  random.randint(30, 89),
            "kyc_status":        "pending",
            "income_monthly":    3000,
            "velocity_baseline": 1.0,
            "balance":           random.uniform(200, 1500),
            "counterparties":    [identity_hash(fake.email()) for _ in range(2)],
            "device_hash":       device_hash(fake.uuid4()),
            "ip_hash":           ip_hash(fake.ipv4()),
            "_txn_count_30d":    random.randint(3, 12),
            "_amount_sum_30d":   random.uniform(300, 1200),
            "_txn_history":      [],
            "has_large_recurring": False,
        }
        txns = []
        for _ in range(random.randint(2, 4)):
            legit_day = max(0, day - random.randint(1, 14))
            txns.append(build_fiat_txn(
                synth, random.choice(synth["counterparties"]),
                sample_amount("student"), rand_ts(legit_day),
                transfer_type=sample_transfer_type(),
                is_fraud=False,
            ))
        inbound_amount   = sample_amount("small_business")
        inbound_ts       = rand_ts(day)
        synth["balance"] += inbound_amount
        drain_amount     = synth["balance"] * random.uniform(0.60, 0.85)
        drain_ts         = inbound_ts + timedelta(hours=random.uniform(2, 12))
        txns.append(build_fiat_txn(
            synth, identity_hash(fake.email()), drain_amount, drain_ts,
            transfer_type=sample_transfer_type(),
            transfer_network=random.choices(["domestic", "international"], weights=[0.60, 0.40])[0],
            is_fraud=True, fraud_type="synthetic_identity",
        ))
        return txns

    def _inject_bec(self, business, day) -> list:
        mule_hash = identity_hash(fake.email())
        amount    = sample_amount("small_business") * random.uniform(1.0, 5.0)
        return [build_fiat_txn(
            business, mule_hash, amount, rand_ts(day),
            transfer_type=sample_transfer_type(),
            transfer_network=random.choices(["domestic", "international"], weights=[0.70, 0.30])[0],
            is_fraud=True, fraud_type="bec",
        )]

    def _inject_structuring(self, user, day) -> list:
        bridge_id = str(uuid.uuid4())
        txns      = []
        total     = 0.0
        for i in range(random.randint(3, 6)):
            amount = sample_amount(user["archetype"]) * random.uniform(0.3, 0.8)
            ts     = rand_ts(max(0, day - (6 - i)))
            txns.append(build_fiat_txn(
                user, identity_hash(fake.email()), amount, ts,
                transfer_type=sample_transfer_type(),
                is_fraud=True, fraud_type="structuring",
                cross_modality_fraud_id=bridge_id,
            ))
            total += amount
        self.cross_modal_bridges.append({
            "bridge_id": bridge_id, "user_hash": user["identity_hash"],
            "amount": total, "ts": rand_ts(day), "pattern": "structuring_to_crypto",
        })
        return txns

    def _inject_reverse_cashout(self, user, day) -> list:
        bridge_id = str(uuid.uuid4())
        amount    = sample_amount("small_business") * random.uniform(2, 8)
        ts        = rand_ts(day)
        txn       = build_fiat_txn(
            user, identity_hash(fake.email()), amount, ts,
            transfer_type=sample_transfer_type(),
            transfer_network=random.choices(["domestic", "international"], weights=[0.50, 0.50])[0],
            is_fraud=True, fraud_type="laundering_fiat_leg",
            cross_modality_fraud_id=bridge_id,
        )
        self.cross_modal_bridges.append({
            "bridge_id": bridge_id, "user_hash": user["identity_hash"],
            "amount": amount * 1.02,
            "ts": ts - timedelta(hours=random.uniform(6, 48)),
            "pattern": "crypto_to_fiat",
        })
        return [txn]

    def run(self) -> list:
        active        = random.sample(self.identities, int(len(self.identities) * 0.98))
        businesses    = [u for u in active if u["archetype"] == "small_business"]
        anomaly_users = set(random.sample(range(len(active)), int(len(active) * 0.05)))
        fraud_budget  = {
            "account_takeover":   int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.40),
            "mule_network":       int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.30),
            "synthetic_identity": int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.15),
            "bec":                int(TARGET_FIAT_TXN * FIAT_FRAUD_RATE * 0.15),
        }
        fraud_injected       = defaultdict(int)
        structuring_injected = 0
        reverse_injected     = 0

        print(f"\nRunning NeoBank simulator ({SIM_DAYS} days, {len(active):,} users)...")
        for day in tqdm(range(SIM_DAYS)):
            for idx, user in enumerate(active):
                cfg        = ARCHETYPES[user["archetype"]]
                is_weekday = (START_TIME + timedelta(days=day)).weekday() < 5
                rate       = cfg["txn_per_day"] if is_weekday else cfg["txn_per_day"] * 0.4
                for _ in range(int(rng.poisson(rate))):
                    self.transactions.append(self._normal_txn(user, day))
                if idx in anomaly_users and day == 15:
                    self.transactions.append(self._micro_anomaly(user, day))

            daily_ato  = fraud_budget["account_takeover"]  // SIM_DAYS
            daily_mule = fraud_budget["mule_network"]       // SIM_DAYS

            for _ in range(daily_ato):
                if fraud_injected["account_takeover"] < fraud_budget["account_takeover"]:
                    self.transactions.extend(self._inject_ato(random.choice(active), day))
                    fraud_injected["account_takeover"] += 1

            for _ in range(max(1, daily_mule // 8)):
                if fraud_injected["mule_network"] < fraud_budget["mule_network"]:
                    pool = random.sample(active, min(15, len(active)))
                    self.transactions.extend(self._inject_mule_network(pool, day))
                    fraud_injected["mule_network"] += 8

            for _ in range(2):
                if fraud_injected["synthetic_identity"] < fraud_budget["synthetic_identity"]:
                    self.transactions.extend(self._inject_synthetic_identity(day))
                    fraud_injected["synthetic_identity"] += 2

            if businesses and fraud_injected["bec"] < fraud_budget["bec"]:
                self.transactions.extend(self._inject_bec(random.choice(businesses), day))
                fraud_injected["bec"] += 1

            if structuring_injected < 15:
                self.transactions.extend(self._inject_structuring(random.choice(active), day))
                structuring_injected += 1

            if day % 3 == 0 and reverse_injected < 10:
                self.transactions.extend(self._inject_reverse_cashout(random.choice(active), day))
                reverse_injected += 1

        return self.transactions


class CryptoExSimulator:
    def __init__(self, identities, bridges):
        self.identities   = identities
        self.bridges      = bridges
        self.transactions = []

    def _normal_txn(self, user, day):
        arch   = CRYPTO_ARCHETYPES[user.get("crypto_archetype", "hodler")]
        lo, hi = arch["amount_range"]
        return build_crypto_txn(
            user, sha256(user["crypto_wallet"] + "sender"), sha256(fake_wallet()),
            random.uniform(lo, hi), rand_ts(day),
            transfer_type=random.choice(["withdrawal", "deposit", "transfer"]),
            receiver_wallet_type=random.choices(
                ["user_wallet", "exchange", "defi_contract"], weights=[0.60, 0.30, 0.10]
            )[0],
        )

    def _inject_mixer(self, user, day):
        amount     = random.uniform(10000, 500000)
        ts         = rand_ts(day)
        network_id = f"mixer_{uuid.uuid4().hex[:8]}"
        mixer_addr = sha256(random.choice(MIXER_ADDRESSES))
        send = build_crypto_txn(
            user, sha256(user["crypto_wallet"]), mixer_addr, amount, ts,
            transfer_type="withdrawal", receiver_wallet_type="mixer",
            known_mixer_interaction=True, receiver_wallet_age_days=9999,
            is_fraud=True, fraud_type="mixer_usage", fraud_network_id=network_id,
        )
        receive = build_crypto_txn(
            user, mixer_addr, sha256(fake_wallet()),
            amount * random.uniform(0.95, 1.0),
            ts + timedelta(hours=random.uniform(2, 48)),
            transfer_type="deposit", receiver_wallet_type="user_wallet",
            known_mixer_interaction=True, receiver_wallet_age_days=0,
            is_fraud=True, fraud_type="mixer_usage", fraud_network_id=network_id,
        )
        return [send, receive]

    def _inject_rug_pull(self, day):
        network_id = f"rugpull_{uuid.uuid4().hex[:8]}"
        contract   = sha256(fake_wallet())
        dev_wallet = sha256(fake_wallet())
        pool_total = 0.0
        txns       = []
        for _ in range(random.randint(10, 50)):
            victim_amount = random.uniform(100, 10000)
            pool_total   += victim_amount
            fake_user = {
                "identity_hash": identity_hash(fake.email()), "crypto_wallet": fake_wallet(),
                "crypto_archetype": "hodler", "account_age_days": random.randint(100, 1000),
                "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
                "_txn_count_30d": random.randint(1, 20), "_amount_sum_30d": random.uniform(500, 10000),
            }
            txns.append(build_crypto_txn(
                fake_user, sha256(fake_wallet()), contract, victim_amount,
                rand_ts(day - random.randint(0, 14)),
                receiver_wallet_type="contract", receiver_wallet_age_days=random.randint(1, 30),
                is_fraud=True, fraud_type="rug_pull", fraud_network_id=network_id,
            ))
        dev_user = {
            "identity_hash": identity_hash(fake.email()), "crypto_wallet": dev_wallet,
            "crypto_archetype": "trader", "account_age_days": random.randint(30, 365),
            "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
            "_txn_count_30d": 0, "_amount_sum_30d": 0,
        }
        txns.append(build_crypto_txn(
            dev_user, contract, dev_wallet, pool_total, rand_ts(day),
            transfer_type="withdrawal", receiver_wallet_type="user_wallet",
            receiver_wallet_age_days=0, is_fraud=True,
            fraud_type="rug_pull", fraud_network_id=network_id,
        ))
        return txns

    def _inject_phishing(self, user, day):
        network_id         = f"phish_{uuid.uuid4().hex[:8]}"
        malicious_contract = sha256(fake_wallet())
        attacker_wallet    = sha256(fake_wallet())
        drain_amount       = user.get("balance", random.uniform(1000, 100000))
        ts                 = rand_ts(day)
        approval = build_crypto_txn(
            user, sha256(user["crypto_wallet"]), malicious_contract, 0.0, ts,
            transfer_type="contract_approval", receiver_wallet_type="contract",
            receiver_wallet_age_days=random.randint(1, 14),
            is_fraud=True, fraud_type="phishing", fraud_network_id=network_id,
        )
        drain = build_crypto_txn(
            user, malicious_contract, attacker_wallet, drain_amount,
            ts + timedelta(seconds=random.randint(10, 120)),
            transfer_type="withdrawal", receiver_wallet_type="user_wallet",
            receiver_wallet_age_days=0, is_fraud=True,
            fraud_type="phishing", fraud_network_id=network_id,
        )
        drain["immediate_withdrawal"] = True
        return [approval, drain]

    def _inject_pig_butchering(self, day):
        network_id     = f"pig_{uuid.uuid4().hex[:8]}"
        scammer_wallet = sha256(fake_wallet())
        txns           = []
        scammer_user   = {
            "identity_hash": identity_hash(fake.email()), "crypto_wallet": scammer_wallet,
            "crypto_archetype": "trader", "account_age_days": random.randint(30, 180),
            "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
            "_txn_count_30d": 0, "_amount_sum_30d": 0,
        }
        for _ in range(random.randint(10, 40)):
            victim_user = {
                "identity_hash": identity_hash(fake.email()), "crypto_wallet": fake_wallet(),
                "crypto_archetype": "hodler", "account_age_days": random.randint(100, 2000),
                "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
                "_txn_count_30d": 0, "_amount_sum_30d": 0,
            }
            base_amount = random.uniform(500, 2000)
            for p in range(random.randint(3, 8)):
                txns.append(build_crypto_txn(
                    victim_user, sha256(victim_user["crypto_wallet"]), scammer_wallet,
                    base_amount * (1.3 ** p),
                    rand_ts(max(0, day - random.randint(0, 30))),
                    receiver_wallet_type="user_wallet",
                    is_fraud=True, fraud_type="pig_butchering", fraud_network_id=network_id,
                ))
        txns.extend(self._inject_mixer(scammer_user, day))
        return txns

    def _process_bridges(self):
        for bridge in self.bridges:
            user = next((u for u in self.identities if u["identity_hash"] == bridge["user_hash"]), None)
            if user is None:
                user = {
                    "identity_hash": bridge["user_hash"], "crypto_wallet": fake_wallet(),
                    "crypto_archetype": "trader", "account_age_days": random.randint(1, 30),
                    "device_hash": device_hash(fake.uuid4()), "ip_hash": ip_hash(fake.ipv4()),
                    "_txn_count_30d": 0, "_amount_sum_30d": 0,
                }
            pattern = bridge["pattern"]
            fiat_ts = bridge["ts"]
            if pattern == "immediate_laundering":
                crypto_ts, delay_label = fiat_ts + timedelta(minutes=random.randint(10, 60)), "immediate_laundering"
            elif pattern == "delayed_laundering":
                crypto_ts, delay_label = fiat_ts + timedelta(hours=random.uniform(24, 72)), "delayed_laundering"
            elif pattern == "structuring_to_crypto":
                crypto_ts, delay_label = fiat_ts + timedelta(hours=random.uniform(1, 12)), "structuring_to_crypto"
            else:
                crypto_ts, delay_label = fiat_ts, "crypto_to_fiat"
            crypto_txn = build_crypto_txn(
                user, sha256(user["crypto_wallet"]), sha256(fake_wallet()),
                bridge["amount"] * random.uniform(0.95, 0.99), crypto_ts,
                transfer_type="withdrawal", receiver_wallet_type="user_wallet",
                receiver_wallet_age_days=0, is_fraud=True,
                fraud_type="laundering_crypto_leg",
                fraud_network_id=bridge.get("network_id"),
                cross_modality_fraud_id=bridge["bridge_id"],
            )
            crypto_txn["cross_modality_pattern"] = delay_label
            crypto_txn["immediate_withdrawal"]   = True
            self.transactions.append(crypto_txn)

    def run(self) -> list:
        active = [u for u in self.identities if u.get("has_crypto", False)]
        if not active:
            active = random.sample(self.identities, int(len(self.identities) * 0.35))
        print(f"\nRunning CryptoEx simulator ({len(active):,} users, {SIM_DAYS} days)...")
        estimated    = len(active) * SIM_DAYS * 0.5
        fraud_budget = int(estimated * CRYPTO_FRAUD_RATE / (1 - CRYPTO_FRAUD_RATE))
        mixer_budget = int(fraud_budget * 0.35)
        rug_budget   = int(fraud_budget * 0.25)
        phish_budget = int(fraud_budget * 0.20)
        pig_budget   = int(fraud_budget * 0.20)
        mixer_count  = rug_count = phish_count = pig_count = 0

        for day in tqdm(range(SIM_DAYS)):
            for user in active:
                arch = CRYPTO_ARCHETYPES[user.get("crypto_archetype", "hodler")]
                if random.random() < arch["txn_per_day"]:
                    self.transactions.append(self._normal_txn(user, day))
            if mixer_count < mixer_budget:
                self.transactions.extend(self._inject_mixer(random.choice(active), day))
                mixer_count += 2
            if rug_count < rug_budget and day % 3 == 0:
                self.transactions.extend(self._inject_rug_pull(day))
                rug_count += 30
            if phish_count < phish_budget:
                self.transactions.extend(self._inject_phishing(random.choice(active), day))
                phish_count += 2
            if pig_count < pig_budget and day % 5 == 0:
                self.transactions.extend(self._inject_pig_butchering(day))
                pig_count += 50

        print("  → Processing cross-modal bridges...")
        self._process_bridges()
        return self.transactions


class FinBridgeSimulator:
    def __init__(self, identities):
        self.identities   = identities
        self.transactions = []

    def run(self) -> list:
        active = random.sample(self.identities, int(len(self.identities) * 0.15))
        print(f"\nRunning FinBridge simulator ({len(active):,} users)...")
        for user in tqdm(active):
            for _ in range(random.randint(1, 5)):
                amount    = sample_amount(user["archetype"])
                day       = random.randint(0, SIM_DAYS - 1)
                ts        = rand_ts(day)
                tx_link   = uuid.uuid4().hex[:8]
                is_fraud  = random.random() < (FIAT_FRAUD_RATE * 1.5)
                bridge_id = str(uuid.uuid4()) if is_fraud else None
                fiat_rec  = build_fiat_txn(
                    user, sha256("finbridge_internal"), amount, ts,
                    transfer_type=sample_transfer_type(), transfer_network="domestic",
                    is_fraud=is_fraud,
                    fraud_type="laundering_fiat_leg" if is_fraud else None,
                    cross_modality_fraud_id=bridge_id,
                )
                fiat_rec["transaction_id"] = f"finbridge_f_{tx_link}"
                fiat_rec["client_id"]      = "finbridge_prod"
                self.transactions.append(fiat_rec)
                crypto_rec = build_crypto_txn(
                    user, sha256(user["crypto_wallet"]), sha256(fake_wallet()),
                    amount * 0.99, ts + timedelta(seconds=30),
                    transfer_type="crypto_purchase", receiver_wallet_type="user_wallet",
                    is_fraud=is_fraud,
                    fraud_type="laundering_crypto_leg" if is_fraud else None,
                    cross_modality_fraud_id=bridge_id,
                )
                crypto_rec["transaction_id"]         = f"finbridge_c_{tx_link}"
                crypto_rec["client_id"]               = "finbridge_prod"
                crypto_rec["linked_fiat_transaction"] = fiat_rec["transaction_id"]
                self.transactions.append(crypto_rec)
        return self.transactions


def validate_and_summarize(df):
    print("\n" + "="*60)
    print("SIMULATION SUMMARY v3.1")
    print("="*60)
    total = len(df)
    fraud = df["is_fraud"].sum()
    print(f"Total: {total:,}  |  Fraud: {fraud:,} ({fraud/total*100:.2f}%)")

    print("\n── By client ──")
    for client, grp in df.groupby("client_id"):
        f = grp["is_fraud"].sum()
        print(f"  {client:<20} {len(grp):>8,}  fraud: {f:>6,} ({f/len(grp)*100:.1f}%)")

    print("\n── Fraud types ──")
    for ft, cnt in df[df["is_fraud"]]["fraud_type"].value_counts().items():
        print(f"  {str(ft):<35} {cnt:>6,}")

    # Distribution overlap checks for the three fixed features
    print("\n── is_first_time_receiver distribution ──")
    fiat_df = df[df["modality"] == "fiat"]
    if len(fiat_df) > 0:
        fraud_ftr = fiat_df[fiat_df["is_fraud"]]["is_first_time_receiver"].mean() * 100
        legit_ftr = fiat_df[~fiat_df["is_fraud"].astype(bool)]["is_first_time_receiver"].mean() * 100
        print(f"  Fraud: {fraud_ftr:.1f}% first-time  |  Legit: {legit_ftr:.1f}% first-time")
        print(f"  Target: Fraud ~55-65%, Legit ~35-45%  (current gap: {fraud_ftr - legit_ftr:.1f}pp)")

    print("\n── balance_drain_ratio distribution (fiat only) ──")
    if len(fiat_df) > 0:
        fraud_drain = fiat_df[fiat_df["is_fraud"]]["balance_drain_ratio"]
        legit_drain = fiat_df[~fiat_df["is_fraud"].astype(bool)]["balance_drain_ratio"]
        print(f"  Fraud: mean={fraud_drain.mean():.3f}  p75={fraud_drain.quantile(0.75):.3f}  p95={fraud_drain.quantile(0.95):.3f}")
        print(f"  Legit: mean={legit_drain.mean():.3f}  p75={legit_drain.quantile(0.75):.3f}  p95={legit_drain.quantile(0.95):.3f}")
        legit_high_drain = (legit_drain > 0.35).mean() * 100
        print(f"  Legit rows with drain > 0.35: {legit_high_drain:.1f}%  (target: 3-5%)")

    print("\n── Cohen's d — top 10 separating features ──")
    print("  (d > 1.0 = artifact, d 0.5-1.0 = moderate, d < 0.5 = realistic)")
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "is_fraud"]
    results      = []
    fraud_df_    = df[df["is_fraud"] == True]
    legit_df_    = df[df["is_fraud"] == False]
    for col in numeric_cols:
        fd, ld = fraud_df_[col].dropna(), legit_df_[col].dropna()
        if len(fd) < 10 or len(ld) < 10:
            continue
        pooled_std = np.sqrt((fd.std()**2 + ld.std()**2) / 2 + 1e-9)
        results.append((col, round(abs(fd.mean() - ld.mean()) / pooled_std, 3)))
    for col, d in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
        flag = " ← ARTIFACT" if d > 1.0 else (" ← moderate" if d > 0.5 else " ✅")
        print(f"  {col:<35} d={d:.3f}{flag}")

    cross = df[df["cross_modality_fraud_id"].notna()]
    print(f"\n── Cross-modal: {len(cross):,} txns, {cross['cross_modality_fraud_id'].nunique():,} sequences ──")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("SENTINEL SIMULATOR v3.1 — Path B (fully patched)")
    print("All three artifact fixes applied:")
    print("  1. is_first_time_receiver: ATO known-payee + mule pre-existing")
    print("  2. balance_drain_ratio: 3% high-drain legitimate transactions")
    print("  3. amount_ratio: repeat_rate * 0.77 → richer user histories")
    print("="*60)

    identities  = build_identity_pool(N_IDENTITIES)

    neobank     = NeoBankSimulator(identities)
    fiat_txns   = neobank.run()
    print(f"  NeoBank:   {len(fiat_txns):,} txns, {len(neobank.cross_modal_bridges):,} bridges")

    finbridge   = FinBridgeSimulator(identities)
    bridge_txns = finbridge.run()
    print(f"  FinBridge: {len(bridge_txns):,} txns")

    cryptoex    = CryptoExSimulator(identities, neobank.cross_modal_bridges)
    crypto_txns = cryptoex.run()
    print(f"  CryptoEx:  {len(crypto_txns):,} txns")

    all_txns = fiat_txns + bridge_txns + crypto_txns
    df       = pd.DataFrame(all_txns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)

    validate_and_summarize(df)

    out_path = "sentinel_training_data.jsonl"
    df.to_json(out_path, orient="records", lines=True, date_format="iso")
    print(f"\nSaved → {out_path}")
    df[df["is_fraud"]].head(1000).to_json("sentinel_fraud_sample.jsonl", orient="records", lines=True)
    print("Saved → sentinel_fraud_sample.jsonl")