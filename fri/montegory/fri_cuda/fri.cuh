#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <string>

#define POSEIDON_DEFINE_CONSTANTS
#include "poseidon.cuh"
#include "util.cuh"

struct FRICommitmentGPU {
    std::vector<std::string> layer_roots;
    std::vector<int>         layer_sizes;
    //std::vector<float>       layer_times;
};

FRICommitmentGPU fri_commitment_gpu(
    uint64_t* d_evals,    // Poseidon은 uint64_t BabyBear 원소 사용
    int initial_size,
    int num_layers,
    cudaStream_t stream = 0
) {
    FRICommitmentGPU result;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    uint64_t* d_buf[2];
    cudaMalloc(&d_buf[0], initial_size * sizeof(uint64_t));
    cudaMalloc(&d_buf[1], initial_size * sizeof(uint64_t));

    cudaMemcpyAsync(d_buf[0], d_evals, initial_size * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    int cur_size = initial_size;
    int src = 0, dst = 1;

    std::cout << "\nFRI Commitment Layers (GPU Poseidon12):" << std::endl;

    for (int layer = 0; layer < num_layers && cur_size > 1; layer++) {
        //cudaEvent_t t0, t1;
        //cudaEventCreate(&t0);
        //cudaEventCreate(&t1);

        //cudaEventRecord(t0, stream);

        int n_pairs = cur_size / 2;
        int grid    = (n_pairs + blockSize - 1) / blockSize;
        poseidon_merkle_layer<<<grid, blockSize, 0, stream>>>(d_buf[src], d_buf[dst], n_pairs);

        //cudaEventRecord(t1, stream);
        //cudaEventSynchronize(t1);

        //float ms = 0.0f;
        //cudaEventElapsedTime(&ms, t0, t1);

        uint64_t root_val = 0;
        if (n_pairs == 1) {
            cudaMemcpyAsync(&root_val, d_buf[dst], sizeof(uint64_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(16) << root_val;
        result.layer_roots.push_back(oss.str());
        result.layer_sizes.push_back(cur_size);
        //result.layer_times.push_back(ms);

        //std::cout << "  Layer " << layer
        //          << ": " << cur_size << " → " << n_pairs
        //          << "  (" << std::fixed << std::setprecision(3) << ms << " ms)";
        //if (n_pairs == 1) std::cout << "  root=" << root_val;
        //std::cout << std::endl;

        std::swap(src, dst);
        cur_size = n_pairs;

        //cudaEventDestroy(t0);
        //cudaEventDestroy(t1);
    }

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    std::cout << "FRI time: " << ms << " ms" << std::endl;
    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
    return result;
}
