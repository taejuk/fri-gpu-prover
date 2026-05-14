#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <string>

#define POSEIDON_DEFINE_CONSTANTS
#include "poseidon.cuh"
#include "util.cuh"
struct FRICommitmentGPU {
  std::vector<std::string> layer_roots;
  std::vector<int> layer_sizes;
  std::vector<float> layer_times;
};

FRICommitmentGPU fri_commitment_gpu(
  uint64_t* d_evals,
  int initial_size,
  int num_layers,
  cudaStream_t stream = 0
) {
  FRICommitmentGPU result;

  uint64_t* d_buf[2];
  cudaMalloc(&d_buf[0], initial_size * sizeof(uint64_t));
  cudaMalloc(&d_buf[1], initial_size * sizeof(uint64_t));

  cudaMemcpy(d_buf[0], d_evals, initial_size * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

  int cur_size = initial_size;
  int src = 0, dst = 1;

  std::cout << "\nFRI Commitment Layers (GPU Poseidon12):" << std::endl;

  for (int layer = 0; layer < num_layers && cur_size > 1; layer++) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0, stream);

    // ── GPU Poseidon Merkle 레이어 ──────────────────────────────────
    // 현재 cur_size 개 노드 → cur_size/2 개 부모 노드
    int n_pairs = cur_size / 2;
    int grid    = (n_pairs + blockSize - 1) / blockSize;
    poseidon_merkle_layer<<<grid, blockSize, 0, stream>>>(d_buf[src], d_buf[dst], n_pairs);
    //cudaDeviceSynchronize();

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    // 루트 확인용: 마지막 레이어면 host 에 가져옴
    uint64_t root_val = 0;
    if (n_pairs == 1) {
        cudaMemcpy(&root_val, d_buf[dst], sizeof(uint64_t),
                    cudaMemcpyDeviceToHost);
    }

    // 루트를 hex 문자열로 저장 (비교 가능하도록)
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << root_val;
    result.layer_roots.push_back(oss.str());
    result.layer_sizes.push_back(cur_size);
    result.layer_times.push_back(ms);

    std::cout << "  Layer " << layer
              << ": " << cur_size << " → " << n_pairs
              << "  (" << std::fixed << std::setprecision(3) << ms << " ms)";
    if (n_pairs == 1)
        std::cout << "  root=" << root_val;
    std::cout << std::endl;

  
    std::swap(src, dst);
    cur_size = n_pairs;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }

  cudaFree(d_buf[0]);
  cudaFree(d_buf[1]);
  return result;
}

