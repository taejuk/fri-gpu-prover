#include <iostream>
#include "util.cuh"
#include "ntt.cuh"
#include "fri.cuh"
#include <chrono>


int main(int argc, char** argv) {
    std::cout << "=== CUDA FRI COMMITMENT ===" << std::endl << std::endl;

    int log_n = std::stoi(argv[1]);
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;
    size_t bytes = n * sizeof(uint64_t);
    int gridSize = (n + blockSize - 1) / blockSize;

    uint64_t* h_coeff;
    cudaMallocHost(&h_coeff, bytes);
    for (uint64_t i = 0; i < n; i++) h_coeff[i] = i % p;
    
    auto start = std::chrono::steady_clock::now();
    
    //cudaStream_t stream_gen, stream_transfer, stream_main;
    //cudaStreamCreate(&stream_gen);
    //cudaStreamCreate(&stream_transfer);
    //cudaStreamCreate(&stream_main);

    cudaError_t poseidon_err = poseidon_init();
    if (poseidon_err != cudaSuccess) {
        std::cerr << "poseidon_init failed: "
                  << cudaGetErrorString(poseidon_err) << std::endl;
        return 1;
    }


    std::cout << "Configuration:" << std::endl;
    std::cout << "  Polynomial size: 2^" << log_n << " = " << n << std::endl;
    std::cout << "  Field modulus: " << p << std::endl;
    std::cout << "  Memory required: " << (bytes / (1024*1024)) << " MB" << std::endl << std::endl;


    uint64_t *d_coeff, *d_twiddles_ntt;
    cudaMalloc(&d_coeff,         bytes);
    cudaMalloc(&d_twiddles_ntt,  bytes);

    uint64_t generator     = get_generator(root, p - 1, n, p);

    generateWithoutInv<<<gridSize, blockSize>>>(d_twiddles_ntt, generator, n, p);
    //cudaMemcpyAsync(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice, stream_transfer);
    cudaMemcpy(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //cudaStreamSynchronize(stream_gen);
    //cudaStreamSynchronize(stream_transfer);

    ntt(d_coeff, d_twiddles_ntt, n, p);

    FRICommitmentGPU fri = fri_commitment_gpu(d_coeff, n, log_n);
    
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "걸린 시간: " << elapsed_seconds.count() << "s\n";

    cudaFreeHost(h_coeff);
    cudaFree(d_coeff);
    cudaFree(d_twiddles_ntt);
    //cudaStreamDestroy(stream_gen);
    //cudaStreamDestroy(stream_transfer);
    //cudaStreamDestroy(stream_main);
    return 0;
}
