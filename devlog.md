
## Day 1 (Phase A start) — Baseline measurement

### Hardware
- GPU: NVIDIA TITAN V (Volta, sm_70, 12 GB HBM2)
- Server: SKKU SW ji cluster
- CUDA 12.6, GCC 13.4

### Baseline (existing fri_cuda)
- Field: BabyBear (p = 2^31 - 2^27 + 1)
- N=2^20: NTT 0.98 ms, IFFT 1.01 ms, FRI commit 48.74 ms

### Key observation
FRI commit (48 ms) ≈ 50× NTT (1 ms). Root cause: CPU↔GPU memcpy + CPU-side
hashing per layer. Primary target for Week 2-3 work.

