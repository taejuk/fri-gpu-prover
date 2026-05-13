#include <cuda_runtime.h>
#include <string>
#include "util.cuh"

#define PAD_FACTOR 4
#define PAD(x) (x + (x >> PAD_FACTOR))


__global__ void bit_reverse_kernel(uint64_t* d_data, int n, int log_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  int rev = 0;
    int temp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }

    if (idx < rev) {
        uint64_t val_i = d_data[idx];
        uint64_t val_r = d_data[rev];
        d_data[idx] = val_r;
        d_data[rev] = val_i;
    }
}

__global__ void ntt_shared_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles, uint64_t n, uint64_t p) {
    __shared__ uint64_t s_data[SHARED_SIZE + (SHARED_SIZE >> PAD_FACTOR)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;

    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    if (block_offset + idx1 < n)
        s_data[PAD(idx1)] = d_data[block_offset + idx1];

    if (block_offset + idx2 < n)
        s_data[PAD(idx2)] = d_data[block_offset + idx2];

    __syncthreads();

    for (int len = 2; len <= SHARED_SIZE; len <<= 1) {
        int pair_dist = len / 2;

        int group = tid / pair_dist;
        int offset = tid % pair_dist;

        int i = group * len + offset;
        int j = i + pair_dist;

        int twiddle_idx = offset * (n / len);
        uint64_t w = d_twiddles[twiddle_idx];

        uint64_t u = s_data[PAD(i)];
        uint64_t v = s_data[PAD(j)];

        uint64_t vw = mul_mod(v, w, p);

        s_data[PAD(i)] = add_mod(u, vw, p);
        s_data[PAD(j)] = sub_mod(u, vw, p);

        __syncthreads();
    }

    if (block_offset + idx1 < n)
        d_data[block_offset + idx1] = s_data[PAD(idx1)];

    if (block_offset + idx2 < n)
        d_data[block_offset + idx2] = s_data[PAD(idx2)];

}

__global__ void ntt_global_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles_inv, int len, uint64_t n, uint64_t p)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid >= n / 2) return;
  int pair_dist = len / 2;
  int group = tid / pair_dist;
  int offset = tid % pair_dist;

  int i = group * len + offset;
  int j = i + pair_dist;

  int twiddle_idx = offset * (n / len);
  uint64_t w_inv = d_twiddles_inv[twiddle_idx];

  uint64_t u = d_data[i];
  uint64_t v = d_data[j];
  uint64_t vw_inv = mul_mod(v, w_inv, p);

  d_data[i] = add_mod(u, vw_inv, p);
  d_data[j] = sub_mod(u, vw_inv, p);
}


void ntt(uint64_t* d_data, uint64_t* d_twiddles, int n, uint64_t p) {
  int gridSize = (n + blockSize - 1) / blockSize;
  int log_n = 0;
  int temp = n;
  while(temp > 1) {
    log_n++;
    temp >>= 1;
  }
  // 이거를 하나의 stream으로 관리하면 된다
  bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
  cudaDeviceSynchronize();

  int shared_grid_size = n / SHARED_SIZE;
  ntt_shared_kernel<<<shared_grid_size, blockSize>>>(d_data, d_twiddles, n, p);
  cudaDeviceSynchronize();

  for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
    gridSize = (n/2 + blockSize - 1) / blockSize;
    ntt_global_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles, len, n, p);
    cudaDeviceSynchronize();
  }
}
