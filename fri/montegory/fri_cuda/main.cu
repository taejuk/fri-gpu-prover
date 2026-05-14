#include <iostream>
#include "util.cuh"
#include "ntt.cuh"
#include "intt.cuh"
#include "fri.cuh"

int main(int argc, char** argv) {
    std::cout << "=== CUDA NTT/INTT with FRI Commitment (Montgomery mul) ===" << std::endl << std::endl;

    int log_n = std::stoi(argv[1]);
    uint64_t p = MONT_P;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;

    cudaStream_t stream_gen, stream_transfer, stream_main;
    cudaStreamCreate(&stream_gen);
    cudaStreamCreate(&stream_transfer);
    cudaStreamCreate(&stream_main);

    // Poseidon 초기화
    cudaError_t poseidon_err = poseidon_init();
    if (poseidon_err != cudaSuccess) {
        std::cerr << "poseidon_init failed: " << cudaGetErrorString(poseidon_err) << std::endl;
        return 1;
    }

    // Poseidon bit-exact 검증
    {
        uint64_t h_in[2] = {0ULL, 0ULL}, h_out = 0;
        uint64_t *d_in, *d_out;
        cudaMalloc(&d_in, 2 * sizeof(uint64_t));
        cudaMalloc(&d_out, sizeof(uint64_t));
        cudaMemcpy(d_in, h_in, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        poseidon_merkle_layer<<<1, 1, 0, stream_main>>>(d_in, d_out, 1);
        cudaStreamSynchronize(stream_main);
        cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "Poseidon bit-exact: compress(0,0) = " << h_out
                  << (h_out == 877077992ULL ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
        cudaFree(d_in); cudaFree(d_out);
    }

    std::cout << "Montgomery: P=0x" << std::hex << MONT_P
              << "  P'=0x" << MONT_PPRIME
              << "  R2=0x" << MONT_R2 << std::dec << std::endl;

    size_t bytes32 = n * sizeof(uint32_t);
    size_t bytes64 = n * sizeof(uint64_t);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Polynomial size: 2^" << log_n << " = " << n << std::endl;
    std::cout << "  Field modulus:   " << p << std::endl;
    std::cout << "  NTT buffer:      " << (bytes32 / (1024*1024)) << " MB (uint32)" << std::endl;

    // Pinned host memory
    uint32_t* h_coeff;
    cudaMallocHost(&h_coeff, bytes32);
    for (uint64_t i = 0; i < n; i++) h_coeff[i] = (uint32_t)(i % p);

    // Device memory
    // d_coeff: NTT/INTT 용 (uint32_t)
    // d_evals: FRI Poseidon 용 (uint64_t)
    uint32_t *d_coeff, *d_twiddles_ntt, *d_twiddles_intt;
    uint64_t *d_evals;
    cudaMalloc(&d_coeff,         bytes32);
    cudaMalloc(&d_twiddles_ntt,  bytes32);
    cudaMalloc(&d_twiddles_intt, bytes32);
    cudaMalloc(&d_evals,         bytes64);

    cudaEvent_t gen_done, transfer_done;
    cudaEventCreate(&gen_done);
    cudaEventCreate(&transfer_done);

    // generator를 호스트에서 계산 후 Montgomery 형태로 변환
    uint64_t generator     = get_generator(root, p - 1, n, p);
    uint64_t generator_inv = mod_inverse_prime(generator, p);
    uint32_t g_m     = to_mont((uint32_t)generator);
    uint32_t g_inv_m = to_mont((uint32_t)generator_inv);

    std::cout << "\nGenerating twiddle factors + transferring data (parallel)..." << std::endl;

    // ── 병렬 Setup ────────────────────────────────────────────
    int gridSize = (n + blockSize - 1) / blockSize;

    // stream_gen: twiddle (Montgomery 형태로 생성)
    generate<<<gridSize, blockSize, 0, stream_gen>>>(
        d_twiddles_ntt, d_twiddles_intt, g_m, g_inv_m, n);
    cudaEventRecord(gen_done, stream_gen);

    // stream_transfer: H2D (pinned → device, PCIe copy 엔진)
    cudaMemcpyAsync(d_coeff, h_coeff, bytes32, cudaMemcpyHostToDevice, stream_transfer);
    cudaEventRecord(transfer_done, stream_transfer);

    // stream_main이 두 stream 완료 후 시작
    cudaStreamWaitEvent(stream_main, gen_done,      0);
    cudaStreamWaitEvent(stream_main, transfer_done, 0);

    cudaEvent_t total_start, total_stop, ntt_start, ntt_stop, fri_start, fri_stop;
    cudaEventCreate(&total_start); cudaEventCreate(&total_stop);
    cudaEventCreate(&ntt_start);   cudaEventCreate(&ntt_stop);
    cudaEventCreate(&fri_start);   cudaEventCreate(&fri_stop);

    cudaEventRecord(total_start, stream_main);

    // ===== NTT =====
    std::cout << "\n--- Forward NTT ---" << std::endl;
    cudaEventRecord(ntt_start, stream_main);

    ntt(d_coeff, d_twiddles_ntt, n, stream_main);

    cudaEventRecord(ntt_stop, stream_main);
    cudaEventSynchronize(ntt_stop);
    float ntt_ms = 0;
    cudaEventElapsedTime(&ntt_ms, ntt_start, ntt_stop);
    std::cout << "NTT time: " << std::fixed << std::setprecision(2) << ntt_ms << " ms" << std::endl;

    // NTT 결과(uint32_t) → d_evals(uint64_t): GPU 커널로 변환
    expand_u32_to_u64<<<gridSize, blockSize, 0, stream_main>>>(d_coeff, d_evals, n);

    // ===== INTT =====
    std::cout << "\n--- Inverse NTT ---" << std::endl;
    cudaEventRecord(ntt_start, stream_main);

    intt(d_coeff, d_twiddles_intt, n, p, stream_main);

    cudaEventRecord(ntt_stop, stream_main);
    cudaEventSynchronize(ntt_stop);
    float intt_ms = 0;
    cudaEventElapsedTime(&intt_ms, ntt_start, ntt_stop);
    std::cout << "INTT time: " << std::fixed << std::setprecision(2) << intt_ms << " ms" << std::endl;

    // ===== FRI Commitment =====
    std::cout << "\n--- FRI Commitment ---" << std::endl;
    cudaEventRecord(fri_start, stream_main);

    FRICommitmentGPU fri = fri_commitment_gpu(d_evals, n, log_n, stream_main);

    cudaEventRecord(fri_stop, stream_main);
    cudaEventSynchronize(fri_stop);
    float fri_ms = 0;
    cudaEventElapsedTime(&fri_ms, fri_start, fri_stop);
    std::cout << "FRI commitment time: " << std::fixed << std::setprecision(2) << fri_ms << " ms" << std::endl;

    cudaEventRecord(total_stop, stream_main);
    cudaEventSynchronize(total_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);

    // ===== Correctness Verification =====
    std::cout << "\n--- Correctness Verification ---" << std::endl;

    uint32_t* h_result;
    cudaMallocHost(&h_result, bytes32);
    cudaStreamSynchronize(stream_main);
    cudaMemcpy(h_result, d_coeff, bytes32, cudaMemcpyDeviceToHost);

    bool all_match = true;
    int error_count = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t expected = (uint32_t)(i % p);
        if (h_result[i] != expected) {
            if (error_count < 5)
                std::cout << "Error at " << i << ": expected " << expected
                          << ", got " << h_result[i] << std::endl;
            error_count++;
            all_match = false;
        }
    }
    if (all_match)
        std::cout << "✓ NTT → INTT verification passed!" << std::endl;
    else
        std::cout << "✗ Verification failed: " << error_count << " mismatches" << std::endl;

    // ===== Performance Summary =====
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::left  << std::setw(25) << "NTT (forward):"
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << ntt_ms  << " ms" << std::endl;
    std::cout << std::left  << std::setw(25) << "INTT (inverse):"
              << std::right << std::setw(10) << intt_ms << " ms" << std::endl;
    std::cout << std::left  << std::setw(25) << "FRI commitment:"
              << std::right << std::setw(10) << fri_ms  << " ms" << std::endl;
    std::cout << std::left  << std::setw(25) << "Total:"
              << std::right << std::setw(10) << total_ms << " ms" << std::endl;
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  NTT:  " << std::fixed << std::setprecision(2)
              << (n * log_n / 1e6 / ntt_ms)  << " GOP/s" << std::endl;
    std::cout << "  INTT: " << (n * log_n / 1e6 / intt_ms) << " GOP/s" << std::endl;

    // Cleanup
    cudaFreeHost(h_coeff);
    cudaFreeHost(h_result);
    cudaFree(d_coeff);
    cudaFree(d_evals);
    cudaFree(d_twiddles_ntt);
    cudaFree(d_twiddles_intt);
    cudaEventDestroy(total_start); cudaEventDestroy(total_stop);
    cudaEventDestroy(ntt_start);   cudaEventDestroy(ntt_stop);
    cudaEventDestroy(fri_start);   cudaEventDestroy(fri_stop);
    cudaEventDestroy(gen_done);
    cudaEventDestroy(transfer_done);
    cudaStreamDestroy(stream_gen);
    cudaStreamDestroy(stream_transfer);
    cudaStreamDestroy(stream_main);

    return 0;
}
