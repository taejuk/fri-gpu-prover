#include <iostream>
#include "util.cuh"
#include "ntt.cuh"
#include "intt.cuh"
#include "fri.cuh"

int main(int argc, char** argv) {
    std::cout << "=== CUDA FFT/IFFT with FRI Commitment ===" << std::endl << std::endl;

    int log_n = std::stoi(argv[1]);
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // Poseidon12 상수를 constant memory 에 로드
    cudaError_t poseidon_err = poseidon_init();
    if (poseidon_err != cudaSuccess) {
        std::cerr << "poseidon_init failed: "
                  << cudaGetErrorString(poseidon_err) << std::endl;
        return 1;
    }

    {
        uint64_t h_in[2]  = {0ULL, 0ULL};
        uint64_t h_out    = 0;
        uint64_t *d_in, *d_out;
        cudaMalloc(&d_in,  2*sizeof(uint64_t));
        cudaMalloc(&d_out,   sizeof(uint64_t));
        cudaMemcpy(d_in, h_in, 2*sizeof(uint64_t), cudaMemcpyHostToDevice);
        poseidon_merkle_layer<<<1,1>>>(d_in, d_out, 1);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "Poseidon bit-exact: compress(0,0) = " << h_out
              << (h_out == 877077992ULL ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
        cudaFree(d_in); cudaFree(d_out);
    }
    std::cout << "Poseidon12 constants loaded." << std::endl;


    size_t bytes = n * sizeof(uint64_t);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Polynomial size: 2^" << log_n << " = " << n << std::endl;
    std::cout << "  Field modulus: " << p << std::endl;
    std::cout << "  Memory required: " << (bytes / (1024*1024)) << " MB" << std::endl << std::endl;

    // Host memory
    uint64_t* h_coeff = (uint64_t*)malloc(bytes);
    for(uint64_t i = 0; i < n; i++) {
        h_coeff[i] = i % p;
    }

    // Device memory
    uint64_t *d_coeff, *d_twiddles_ntt, *d_twiddles_intt, *d_evals;
    cudaMalloc(&d_coeff, bytes);
    cudaMalloc(&d_twiddles_ntt, bytes);
    cudaMalloc(&d_twiddles_intt, bytes);
    cudaMalloc(&d_evals, bytes);

    // Generate twiddle factors
    uint64_t generator = get_generator(root, p - 1, n, p);
    uint64_t generator_inv = mod_inverse_prime(generator, p);
    std::cout << "generator: " << generator << " " << generator_inv << std::endl;
    std::cout << "Generating twiddle factors..." << std::endl;
    int gridSize = (n + blockSize - 1) / blockSize;
    generate<<<gridSize, blockSize,0, stream>>>(d_twiddles_ntt, d_twiddles_intt, generator, generator_inv, n, p);
    //cudaDeviceSynchronize();


    // Transfer data to device
    cudaMemcpy(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_evals, h_coeff, bytes, cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t total_start, total_stop,ntt_start, ntt_stop, fri_start, fri_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&ntt_start);
    cudaEventCreate(&ntt_stop);
    cudaEventCreate(&fri_start);
    cudaEventCreate(&fri_stop);

    cudaEventRecord(total_start, stream);

    // ===== NTT =====
    std::cout << "\n--- Forward NTT ---" << std::endl;
    cudaEventRecord(ntt_start, stream);

    ntt(d_coeff, d_twiddles_ntt, n, p, stream);

    cudaEventRecord(ntt_stop, stream);
    cudaEventSynchronize(ntt_stop);
    float ntt_ms = 0;
    cudaEventElapsedTime(&ntt_ms, ntt_start, ntt_stop);
    std::cout << "FFT time: " << std::fixed << std::setprecision(2) << ntt_ms << " ms" << std::endl;

    // Copy evaluations
    //cudaMemcpy(d_evals, d_coeff, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_evals, d_coeff, bytes, cudaMemcpyDeviceToDevice, stream);

    // ===== INTT =====
    std::cout << "\n--- Inverse NTT ---" << std::endl;
    cudaEventRecord(ntt_start, stream);

    intt(d_coeff, d_twiddles_intt, n, p, stream);

    cudaEventRecord(ntt_stop, stream);
    cudaEventSynchronize(ntt_stop);
    float intt_ms = 0;
    cudaEventElapsedTime(&intt_ms, ntt_start, ntt_stop);

    std::cout << "IFFT time: " << std::fixed << std::setprecision(2) << intt_ms << " ms" << std::endl;

    // ===== FRI Commitment =====
    std::cout << "\n--- FRI Commitment ---" << std::endl;
    cudaEventRecord(fri_start, stream);

    FRICommitmentGPU fri = fri_commitment_gpu(d_evals, n, log_n, stream);

    cudaEventRecord(fri_stop, stream);
    cudaEventSynchronize(fri_stop);
    float fri_ms = 0;
    cudaEventElapsedTime(&fri_ms, fri_start, fri_stop);
    std::cout << "FRI commitment time: " << std::fixed << std::setprecision(2) << fri_ms << " ms" << std::endl;

    cudaEventRecord(total_stop, stream);
    cudaEventSynchronize(total_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);

    // ===== Verification =====
    std::cout << "\n--- Correctness Verification ---" << std::endl;

    uint64_t* h_result = (uint64_t*)malloc(bytes);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_result, d_coeff, bytes, cudaMemcpyDeviceToHost);

    bool all_match = true;
    int error_count = 0;
    for(uint64_t i = 0; i < n; i++) {
        uint64_t expected = i % p;
        if(h_result[i] != expected) {
            if(error_count < 5) {
                std::cout << "Error at " << i << ": expected " << expected
                          << ", got " << h_result[i] << std::endl;
            }
            error_count++;
            all_match = false;
        }
    }

    if(all_match) {
        std::cout << "✓ NTT → INTT verification passed! All values match." << std::endl;
    } else {
        std::cout << "✗ Verification failed: " << error_count << " mismatches" << std::endl;
    }

    // ===== Performance Summary =====
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::left << std::setw(25) << "NTT (forward):"
              << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << ntt_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "INTT (inverse):"
              << std::right << std::setw(10) << intt_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "FRI commitment:"
              << std::right << std::setw(10) << fri_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "Total:"
              << std::right << std::setw(10) << total_ms << " ms" << std::endl;

    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  NTT: " << std::fixed << std::setprecision(2)
              << (n * log_n / 1e6 / ntt_ms) << " GOP/s" << std::endl;
    std::cout << "  INTT: " << (n * log_n / 1e6 / intt_ms) << " GOP/s" << std::endl;

    // Cleanup
    free(h_coeff);
    free(h_result);
    cudaFree(d_coeff);
    cudaFree(d_evals);
    cudaFree(d_twiddles_ntt);
    cudaFree(d_twiddles_intt);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(ntt_start);
    cudaEventDestroy(ntt_stop);
    cudaEventDestroy(fri_start);
    cudaEventDestroy(fri_stop);
    cudaStreamDestroy(stream);

    return 0;
}
