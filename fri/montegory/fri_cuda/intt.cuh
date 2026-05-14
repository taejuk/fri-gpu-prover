#include <cuda_runtime.h>
#include "util.cuh"

#define PAD_FACTOR 4
#define PAD(x) (x + (x >> PAD_FACTOR))

__global__ void intt_shared_kernel(uint32_t* d_data,
                                    const uint32_t* __restrict__ d_twiddles_inv,
                                    uint32_t n) {
    __shared__ uint32_t s_data[SHARED_SIZE + (SHARED_SIZE >> PAD_FACTOR)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;

    if (block_offset + tid              < n) s_data[PAD(tid)]              = d_data[block_offset + tid];
    if (block_offset + tid + blockDim.x < n) s_data[PAD(tid + blockDim.x)] = d_data[block_offset + tid + blockDim.x];
    __syncthreads();

    for (int len = 2; len <= SHARED_SIZE; len <<= 1) {
        int pair_dist = len / 2;
        int group  = tid / pair_dist;
        int offset = tid % pair_dist;
        int i = group * len + offset;
        int j = i + pair_dist;

        uint32_t w_inv = d_twiddles_inv[offset * (n / len)];
        uint32_t u     = s_data[PAD(i)];
        uint32_t v     = s_data[PAD(j)];
        uint32_t vw    = mul_mod(v, w_inv);

        s_data[PAD(i)] = add_mod(u, vw);
        s_data[PAD(j)] = sub_mod(u, vw);
        __syncthreads();
    }

    if (block_offset + tid              < n) d_data[block_offset + tid]              = s_data[PAD(tid)];
    if (block_offset + tid + blockDim.x < n) d_data[block_offset + tid + blockDim.x] = s_data[PAD(tid + blockDim.x)];
}

__global__ void intt_global_kernel(uint32_t* d_data,
                                    const uint32_t* __restrict__ d_twiddles_inv,
                                    int len, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    int pair_dist = len / 2;
    int group  = tid / pair_dist;
    int offset = tid % pair_dist;
    int i = group * len + offset;
    int j = i + pair_dist;

    uint32_t w_inv = d_twiddles_inv[offset * (n / len)];
    uint32_t u     = d_data[i];
    uint32_t v     = d_data[j];
    uint32_t vw    = mul_mod(v, w_inv);

    d_data[i] = add_mod(u, vw);
    d_data[j] = sub_mod(u, vw);
}

// n^{-1} mod p 를 Montgomery 형태로 저장
__global__ void intt_scale(uint32_t* d_data, int n, uint32_t n_inv_m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    d_data[tid] = mul_mod(d_data[tid], n_inv_m);
}

void intt(uint32_t* d_data, uint32_t* d_twiddles_inv,
          int n, uint64_t p, cudaStream_t stream = 0) {
    int log_n = 0, temp = n;
    while (temp > 1) { log_n++; temp >>= 1; }

    int gridSize = (n + blockSize - 1) / blockSize;

    // 입력을 Montgomery domain으로 변환
    to_mont_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n);

    bit_reverse_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n, log_n);

    int shared_grid = n / SHARED_SIZE;
    intt_shared_kernel<<<shared_grid, blockSize, 0, stream>>>(d_data, d_twiddles_inv, n);

    for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        intt_global_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, d_twiddles_inv, len, n);
    }

    // n^{-1} scaling — Montgomery 형태로 전달
    uint32_t n_inv   = (uint32_t)mod_inverse_prime(n, p);
    uint32_t n_inv_m = to_mont(n_inv);   // 호스트에서 변환
    gridSize = (n + blockSize - 1) / blockSize;
    intt_scale<<<gridSize, blockSize, 0, stream>>>(d_data, n, n_inv_m);

    // 결과를 일반 값으로 변환
    from_mont_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n);
}
