# Profile Report

## Day 2 — Initial baseline analysis

Measurements: TITAN V, BabyBear (p = 2^31 − 2^27 + 1), 3-trial median.
See [bench_results/own_summary.md](../bench_results/own_summary.md) for raw table.

### Observation 1: NTT underutilizes GPU at small N

NTT timing scales as expected for O(N log N) at N ≥ 2^20 (4.87×–4.92× for 4× N),
but only 1.47× for 2^16 → 2^18. This indicates kernel launch overhead and
GPU underutilization at small N — 65,536 elements cannot saturate the
5,120 CUDA cores of TITAN V. Memory pool (Week 4) and Montgomery
multiplication (Week 5) are expected to disproportionately help small-N cases.

### Observation 2: FRI commit is consistently 25–30× NTT — CPU-bound

The ratio FRI_commit / NTT stays nearly constant (17.8× to 30.5×) across all
tested N. Combined with the fact that the per-layer code in `fri_commitment_gpu`
includes:
- `cudaMemcpy` device→host per layer (8 bytes × N transferred each layer)
- CPU-side `simple_hash64` over N elements

this confirms the bottleneck is CPU work scaling linearly with N, not GPU
compute. Moving Poseidon hashing to GPU (Week 2) and removing
`cudaDeviceSynchronize` between layers (Week 3) should be able to push
the FRI/NTT ratio from ~25× down toward 3–5× — yielding an estimated
4–6× speedup on N=2^24 FRI commit (558ms → ~100ms).

### Implications for Weeks 2–5

| Week | Optimization | Expected primary effect |
|---|---|---|
| 2 | Real GPU Poseidon | Removes CPU hashing per layer |
| 3 | Async streams + remove sync | Removes pipeline stalls between layers |
| 4 | Pinned memory + memory pool | Helps small N (launch overhead dominant) |
| 5 | Montgomery multiplication | Helps NTT inner loop (mul-heavy) |

Cumulative target on N=2^24:
- Baseline: NTT 22.3 ms, FRI commit 558.7 ms, Total 603.6 ms
- Week 5 end (estimated): NTT 8–12 ms, FRI commit 80–120 ms, Total 100–140 ms
- ~5× end-to-end speedup expected.

## Day 3 — Variance check

10-trial variance at N=2^20:
- **NTT: 0.92–0.96 ms** (range/median = 4.3%). Measurement noise within expected.
- **FRI commit: 28.14–37.12 ms** (range/median = 25.5%, bimodal — fast cluster ~28 ms, slow cluster ~36 ms).

The bimodal FRI commit distribution suggests CPU contention from shared-server
co-tenants, consistent with the hypothesis that the bottleneck is CPU-side
hashing inside the FRI commit loop. NTT, being pure GPU work, is unaffected.

**Implication:** The CPU-dominated FRI commit phase is not only slow but *also
unstable*. Moving Poseidon to GPU (Week 2) and removing CPU↔GPU per-layer
ping-pong (Week 3) is expected to both reduce wall-clock time AND eliminate
this variance entirely.

**Revised baseline (using 10-trial median):**
- N=2^20: NTT 0.94 ms, FRI commit 35.26 ms, Total ~37 ms

