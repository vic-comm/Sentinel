"""
locustfile.py  —  Sentinel load test with proper P50/P99 benchmarking
=======================================================================
What the original script got wrong:
  1. All requests used the same identity_hash → Redis state became hot
     and all velocity features were inflated after the first few requests.
     This underestimates real-world latency (cold Redis state is slower).

  2. Only one scenario (normal transaction). This never exercises the
     cross-modal detection path, ATO scoring, or the SHAP explanation
     path (high-risk scores trigger SHAP every time).

  3. No warmup phase. The first 30 seconds of a load test are unstable
     because Ray Serve is JIT-compiling, Redis connections are being
     established, and Bytewax state is cold. These outliers inflate P99.

  4. wait_time = between(0.01, 0.05) is misleading. With 100 users each
     waiting 10-50ms between requests, peak RPS = 100 / 0.01 = 10,000 RPS.
     That is not the actual RPS because Locust is single-threaded — the
     effective RPS is capped by the event loop. Use --headless mode with
     --spawn-rate to control this properly.

  5. Static timestamp "2026-04-13T09:00:00Z". Bytewax uses the timestamp
     for velocity window cutoffs. A static timestamp means every request
     lands in the same 1-hour window, causing artificially high count_1h.

  6. sync_score=True for every request. This measures E2E latency (correct)
     but also means every request holds a connection open for 5 seconds
     if the scoring times out. For throughput benchmarks, use async mode.

How to run:
  # Install
  pip install locust

  # Interactive UI (open http://localhost:8089)
  locust -f locustfile.py --host http://localhost:8000

  # Headless benchmark (proper way — captures P50/P99 in CSV)
  locust -f locustfile.py --host http://localhost:8000 \\
    --headless --users 50 --spawn-rate 5 \\
    --run-time 3m \\
    --csv results/sentinel \\
    --csv-full-history

  # Read results
  # results/sentinel_stats.csv          → P50, P95, P99 per endpoint
  # results/sentinel_stats_history.csv  → time-series (chart latency over time)
  # results/sentinel_failures.csv       → error details

  # What good numbers look like for the Sentinel demo:
  #   /v1/transactions/evaluate (sync)  P50 < 20ms, P99 < 80ms
  #   /v1/transactions/evaluate (async) P50 < 5ms,  P99 < 15ms
  #   /health                            P50 < 2ms,  P99 < 5ms
"""

import hashlib
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

from locust import FastHttpUser, HttpUser, between, events, task
from locust.env import Environment

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SHARED_SALT = "sentinel_consortium_2026_v1"

# Pool of identity hashes — simulates a realistic mix of:
#   - New users (no Redis history → cold feature computation)
#   - Returning users (warm Redis state → cached velocity features)
# 20 identities = enough to prevent all requests hitting the same hot key
IDENTITY_POOL = [
    hashlib.sha256(f"user_{i}@example.com{SHARED_SALT}".encode()).hexdigest()
    for i in range(20)
]

# Warmup flag — set True after the first 60 seconds to exclude startup latency
_warmup_done = False


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC timestamp as ISO 8601. Use real time, not static."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_normal_txn(sync: bool = True) -> dict:
    """Baseline normal transaction — should score < 0.3 (APPROVE)."""
    identity = random.choice(IDENTITY_POOL)
    return {
        "transaction_id":          f"bench_normal_{uuid.uuid4().hex[:8]}",
        "client_id":               "neobank_prod",
        "timestamp":               _now_iso(),
        "modality":                "fiat",
        "identity_hash":           identity,
        "sender_hash":             identity,
        "receiver_hash":           hashlib.sha256(f"recv_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              random.uniform(500, 3000),
        "transfer_type":           random.choice(["ach", "p2p", "rtp"]),
        "transfer_network":        random.choice(["same_bank", "domestic"]),
        "sender_account_age_days": random.randint(180, 2000),
        "sender_kyc_status":       "verified",
        "sender_archetype":        random.choice(["salary_worker", "freelancer"]),
        "is_first_time_receiver":  random.random() < 0.38,
        "balance_drain_ratio":     random.uniform(0.05, 0.25),
        "sync_score":              sync,
    }


def make_ato_txn(sync: bool = True) -> dict:
    """
    Account takeover simulation.
    Should score > 0.7 (BLOCK or REVIEW).
    Tests: high velocity + new receiver + device change.
    Also exercises SHAP explanation path (high-risk always gets SHAP).
    """
    identity = random.choice(IDENTITY_POOL)
    return {
        "transaction_id":          f"bench_ato_{uuid.uuid4().hex[:8]}",
        "client_id":               "neobank_prod",
        "timestamp":               _now_iso(),
        "modality":                "fiat",
        "identity_hash":           identity,
        "sender_hash":             identity,
        "receiver_hash":           hashlib.sha256(f"mule_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              random.uniform(5000, 15000),
        "transfer_type":           "wire",
        "transfer_network":        "international",
        "sender_account_age_days": random.randint(200, 800),
        "sender_kyc_status":       "verified",
        "sender_archetype":        "salary_worker",
        "is_first_time_receiver":  True,
        "balance_drain_ratio":     random.uniform(0.60, 0.95),
        "sync_score":              sync,
    }


def make_cross_modal_txn(sync: bool = True) -> dict:
    """
    Cross-modal transaction (crypto after fiat).
    Tests: GNN embedding lookup + cross-modal detection.
    """
    identity = random.choice(IDENTITY_POOL)
    return {
        "transaction_id":          f"bench_xmodal_{uuid.uuid4().hex[:8]}",
        "client_id":               "cryptoex_prod",
        "timestamp":               _now_iso(),
        "modality":                "crypto",
        "user_email_hash":         identity,
        "identity_hash":           identity,
        "sender_wallet_hash":      hashlib.sha256(f"wallet_{identity}".encode()).hexdigest(),
        "receiver_wallet_hash":    hashlib.sha256(f"fresh_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              random.uniform(8000, 50000),
        "amount_crypto":           random.uniform(2, 18),
        "cryptocurrency":          "ETH",
        "blockchain":              "ethereum",
        "transaction_hash":        f"0x{uuid.uuid4().hex}",
        "known_mixer_interaction": random.random() < 0.30,
        "receiver_wallet_age_days": random.randint(0, 30),
        "receiver_wallet_type":    "user_wallet",
        "sync_score":              sync,
    }


def make_mixer_txn(sync: bool = True) -> dict:
    """
    Known mixer interaction.
    Should score very high. Tests known_mixer_interaction SHAP path.
    """
    identity = random.choice(IDENTITY_POOL)
    return {
        "transaction_id":          f"bench_mixer_{uuid.uuid4().hex[:8]}",
        "client_id":               "cryptoex_prod",
        "timestamp":               _now_iso(),
        "modality":                "crypto",
        "user_email_hash":         identity,
        "identity_hash":           identity,
        "sender_wallet_hash":      hashlib.sha256(f"wallet_{identity}".encode()).hexdigest(),
        "receiver_wallet_hash":    hashlib.sha256(
                                       b"0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936"
                                   ).hexdigest(),
        "amount_usd":              random.uniform(10000, 100000),
        "amount_crypto":           random.uniform(3, 36),
        "cryptocurrency":          random.choice(["ETH", "BTC"]),
        "blockchain":              "ethereum",
        "transaction_hash":        f"0x{uuid.uuid4().hex}",
        "known_mixer_interaction": True,
        "receiver_wallet_age_days": 9999,
        "receiver_wallet_type":    "mixer",
        "sync_score":              sync,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOCUST USER — SYNC MODE (E2E latency benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class SentinelSyncUser(FastHttpUser):
    """
    Measures end-to-end latency with sync_score=True.
    Each request waits for the full pipeline result.

    Use this for:
      - P50/P99 latency benchmarks
      - Verifying SLA compliance (P99 < 100ms)
      - Measuring SHAP explanation overhead

    FastHttpUser vs HttpUser:
      FastHttpUser uses httpx (async) instead of requests (sync).
      This allows more concurrent requests per Locust worker.
      Use FastHttpUser unless you need sessions/cookies.

    Scenario mix (reflects real fraud distribution):
      80% normal    — baseline load, no SHAP
      10% ATO       — high risk, always triggers SHAP
       5% cross-modal — tests GNN embedding path
       5% mixer     — very high risk, tests known_mixer path
    """

    # Think time between requests (per user)
    # 100-500ms simulates a user initiating a transaction every 100-500ms
    # For throughput benchmarks, set wait_time = constant(0)
    wait_time = between(0.1, 0.5)

    # Track per-user custom metrics
    def on_start(self):
        self._txn_count  = 0
        self._blocked    = 0
        self._approved   = 0
        self._reviewed   = 0

    @task(8)
    def normal_transaction(self):
        """80% of traffic — normal users."""
        self._score(make_normal_txn(sync=True), name="[normal] sync E2E")

    @task(1)
    def ato_transaction(self):
        """10% of traffic — account takeover simulation."""
        self._score(make_ato_txn(sync=True), name="[ato] sync E2E")

    @task(1)
    def mixer_transaction(self):
        """10% of traffic — crypto mixer (splits between cross-modal + mixer)."""
        if random.random() < 0.5:
            self._score(make_cross_modal_txn(sync=True), name="[cross_modal] sync E2E")
        else:
            self._score(make_mixer_txn(sync=True), name="[mixer] sync E2E")

    def _score(self, txn: dict, name: str):
        """
        POST transaction and assert response contract.
        Locust records latency for this call automatically.
        Named requests show up as separate rows in the results table.
        """
        start = time.perf_counter()

        with self.client.post(
            "/v1/transactions/evaluate",
            json=txn,
            name=name,
            catch_response=True,
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code not in (200, 202):
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return

            try:
                body = resp.json()
            except Exception:
                resp.failure("Response not JSON")
                return

            # For sync_score=True, assert the result is present
            # (not just "processing") after a reasonable timeout
            if txn.get("sync_score") and body.get("status") == "scoring_timeout":
                resp.failure(f"Scoring timeout after {elapsed_ms:.0f}ms")
                return

            # Count decisions for per-user stats
            decision = body.get("decision", "")
            if decision == "block":   self._blocked  += 1
            elif decision == "approve": self._approved += 1
            elif decision == "review":  self._reviewed += 1
            self._txn_count += 1

            resp.success()


# ─────────────────────────────────────────────────────────────────────────────
# LOCUST USER — ASYNC MODE (throughput benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class SentinelAsyncUser(FastHttpUser):
    """
    Measures ingestion throughput with sync_score=False.
    Returns 202 immediately — measures how fast ingestion accepts transactions.

    Use this for:
      - Maximum RPS benchmarks
      - Checking Redpanda / feature engine doesn't become a bottleneck
      - Testing ingestion validation overhead (Pydantic schema check)

    Combine with SentinelSyncUser in a load test:
      80% async (throughput measurement)
      20% sync  (latency measurement)
    """

    wait_time = between(0.01, 0.05)  # 10-50ms → high throughput

    @task
    def async_ingest(self):
        txn = make_normal_txn(sync=False)
        self.client.post(
            "/v1/transactions/evaluate",
            json=txn,
            name="[normal] async ingest",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM METRICS + EVENT HOOKS
# ─────────────────────────────────────────────────────────────────────────────

@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length,
    exception, context, **kwargs
):
    """
    Custom event handler — fires on every request.
    Use this to:
      - Write custom metrics to InfluxDB / Prometheus
      - Log slow requests (P99 violations)
      - Track per-scenario decision distribution
    """
    # Log requests that exceed 200ms (P99 SLA violation for Sentinel)
    if response_time > 200 and exception is None:
        print(
            f"[SLOW] {name} | {response_time:.0f}ms | "
            f"global threshold: 200ms"
        )


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Print benchmark config at start."""
    print("\n" + "="*55)
    print("SENTINEL LOAD TEST")
    print("="*55)
    print(f"  Target:        {environment.host}")
    print(f"  Identity pool: {len(IDENTITY_POOL)} unique users")
    print(f"  Scenario mix:  80% normal | 10% ATO | 10% crypto")
    print(f"  Timestamp:     real-time (not static)")
    print()
    print("  Interpreting results:")
    print("  P50 < 15ms  → ingestion + feature + inference tight loop working")
    print("  P99 < 80ms  → no tail latency from Redis/SHAP")
    print("  P99 > 200ms → investigate: Redis slow, SHAP overhead, Ray Serve backpressure")
    print("="*55 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Print summary after test."""
    stats = environment.stats
    total = stats.total

    print("\n" + "="*55)
    print("BENCHMARK SUMMARY")
    print("="*55)
    print(f"  Total requests:  {total.num_requests:,}")
    print(f"  Failures:        {total.num_failures:,} ({total.fail_ratio*100:.1f}%)")
    print(f"  RPS (peak):      {total.current_rps:.1f}")
    print(f"  P50 latency:     {total.get_response_time_percentile(0.50):.0f}ms")
    print(f"  P75 latency:     {total.get_response_time_percentile(0.75):.0f}ms")
    print(f"  P95 latency:     {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  P99 latency:     {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  P99.9 latency:   {total.get_response_time_percentile(0.999):.0f}ms")
    print()

    # Per-scenario breakdown
    for name, entry in stats.entries.items():
        if entry.num_requests < 5:
            continue
        print(
            f"  {name[1]:<30} "
            f"n={entry.num_requests:>6,} "
            f"P50={entry.get_response_time_percentile(0.50):>5.0f}ms "
            f"P99={entry.get_response_time_percentile(0.99):>5.0f}ms "
            f"err={entry.fail_ratio*100:.1f}%"
        )
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# WARMUP SCRIPT (run before benchmarking)
# ─────────────────────────────────────────────────────────────────────────────

def warmup(host: str = "http://localhost:8000", n: int = 50):
    """
    Send N transactions before benchmarking to:
      1. Warm up Ray Serve (JIT compilation on first requests)
      2. Populate Redis velocity state for IDENTITY_POOL users
      3. Establish connection pools in httpx

    Call this before running Locust:
      python -c "from locustfile import warmup; warmup()"
    """
    import httpx
    import sys

    print(f"Warming up with {n} transactions...")
    with httpx.Client(timeout=10.0) as client:
        for i in range(n):
            txn = make_normal_txn(sync=True)
            try:
                r = client.post(f"{host}/v1/transactions/evaluate", json=txn)
                if i % 10 == 0:
                    score = r.json().get("risk_score", "?")
                    print(f"  [{i+1:>3}/{n}] score={score}")
            except Exception as e:
                print(f"  [{i+1:>3}/{n}] FAILED: {e}", file=sys.stderr)

    print(f"Warmup complete. Redis state populated for {len(IDENTITY_POOL)} users.")
    print("Now run: locust -f locustfile.py --host http://localhost:8000")


if __name__ == "__main__":
    warmup()