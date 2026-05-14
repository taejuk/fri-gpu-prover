#include <iostream>
#include "util.cuh"
#include "ntt.cuh"
#include "fri.cuh"
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "=== CUDA FRI COMMITMENT (no stream) ===" << std::endl << std::endl;

    int log_n = std::stoi(argv[1]);
    uint64_t p = MONT_P;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;
    size_t bytes32 = n * sizeof(uint32_t);
    size_t bytes64 = n * sizeof(uint64_t);
    int gridSize = (n + blockSize - 1) / blockSize;

    // Pinned host memory (uint32_t)
    uint32_t* h_coeff;
    cudaMallocHost(&h_coeff, bytes32);
    for (uint64_t i = 0; i < n; i++) h_coeff[i] = (uint32_t)(i % p);

    auto start = std::chrono::steady_clock::now();

    cudaError_t poseidon_err = poseidon_init();
    if (poseidon_err != cudaSuccess) {
        std::cerr << "poseidon_init failed: "
                  << cudaGetErrorString(poseidon_err) << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Polynomial size: 2^" << log_n << " = " << n << std::endl;
    std::cout << "  Field modulus: " << p << std::endl;
    std::cout << "  Memory required: " << (bytes32 / (1024*1024)) << " MB" << std::endl << std::endl;

    // Device memory
    uint32_t *d_coeff, *d_twiddles_ntt;
    uint64_t *d_evals;
    cudaMalloc(&d_coeff,        bytes32);
    cudaMalloc(&d_twiddles_ntt, bytes32);
    cudaMalloc(&d_evals,        bytes64);

    // generator → Montgomery 형태로 변환
    uint64_t generator = get_generator(root, p - 1, n, p);
    uint32_t g_m = to_mont((uint32_t)generator);

    // ── 순차 실행 (stream 없음) ───────────────────────────────
    generateWithoutInv<<<gridSize, blockSize>>>(d_twiddles_ntt, g_m, n);
    cudaMemcpy(d_coeff, h_coeff, bytes32, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ── NTT ──────────────────────────────────────────────────
    ntt(d_coeff, d_twiddles_ntt, n);   // stream 없음 → default stream(0)

    // NTT 결과(uint32_t) → FRI 입력(uint64_t) 변환
    expand_u32_to_u64<<<gridSize, blockSize>>>(d_coeff, d_evals, n);
    cudaDeviceSynchronize();

    // ── FRI Commitment ────────────────────────────────────────
    FRICommitmentGPU fri = fri_commitment_gpu(d_evals, n, log_n);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "걸린 시간: " << elapsed.count() << "s\n";

    // Cleanup
    cudaFreeHost(h_coeff);
    cudaFree(d_coeff);
    cudaFree(d_evals);
    cudaFree(d_twiddles_ntt);
    return 0;
}
