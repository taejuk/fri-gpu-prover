# Own Baseline Sweep (TITAN V, BabyBear)

Date: 2026-05-11
GPU: NVIDIA TITAN V (Volta sm_70, 12GB HBM2)
CUDA: 12.6, GCC 13.4
Trials per N: 3 (median reported)

| log_n | N | NTT (ms) | IFFT (ms) | FRI commit (ms) | Total (ms) |
|---|---|---|---|---|---|
| 2^16 | 65536 | 0.17 | 0.18 | 3.03 | 3.49 |
| 2^18 | 262144 | 0.25 | 0.27 | 6.90 | 7.50 |
| 2^20 | 1048576 | 0.93 | 0.99 | 28.32 | 30.31 |
| 2^22 | 4194304 | 4.58 | 4.71 | 112.19 | 121.67 |
| 2^24 | 16777216 | 22.32 | 22.59 | 558.74 | 603.59 |
**Note (Day 3):** N=2^20 FRI commit varies 25.5% bimodal due to shared-server
CPU contention. The 10-trial median is 35.26 ms (vs 28.32 ms here from a
single 3-trial median). Variance is expected to disappear once Week 2-3
work moves hashing to GPU. See variance_analysis.md.
