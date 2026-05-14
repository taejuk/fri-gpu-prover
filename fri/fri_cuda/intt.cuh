#include <cuda_runtime.h>
#include <string>
#include "util.cuh"

#define PAD_FACTOR 4
#define PAD(x) (x + (x >> PAD_FACTOR))

__global__ void intt_shared_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles_inv, int n, uint64_t p) {
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
    uint64_t w_inv = d_twiddles_inv[twiddle_idx];

    uint64_t u = s_data[PAD(i)];
    uint64_t v = s_data[PAD(j)];
    uint64_t vw_inv = mul_mod(v, w_inv, p);

    s_data[PAD(i)] = add_mod(u, vw_inv, p);
    s_data[PAD(j)] = sub_mod(u, vw_inv, p);

    __syncthreads();
  }

  if (block_offset + idx1 < n)
    d_data[block_offset + idx1] = s_data[PAD(idx1)];
      
  if (block_offset + idx2 < n)
    d_data[block_offset + idx2] = s_data[PAD(idx2)];
}

__global__ void intt_global_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles_inv, int len, int n, uint64_t p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= n / 2) return;

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

__global__ void intt_scale(uint64_t* d_data, int n, uint64_t n_inverse, uint64_t p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    d_data[tid] = mul_mod(d_data[tid], n_inverse, p);
}

void intt(uint64_t* d_data, uint64_t* d_twiddles_inv, int n, uint64_t p, cudaStream_t stream = 0) {
  int gridSize = (n + blockSize - 1) / blockSize;
  int log_n = 0;
  int temp = n;
  while(temp > 1) { log_n++; temp >>= 1; }
  
  bit_reverse_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n, log_n);
  //cudaDeviceSynchronize();
  
  int shared_grid_size = n / SHARED_SIZE;
  intt_shared_kernel<<<shared_grid_size, blockSize,0, stream>>>(d_data, d_twiddles_inv, n, p);
  //cudaDeviceSynchronize();
  
  for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
    gridSize = (n / 2 + blockSize - 1) / blockSize;
    intt_global_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, d_twiddles_inv, len, n, p);
    //cudaDeviceSynchronize();
  }
  
  uint64_t n_inverse = mod_inverse_prime(n, p);
  gridSize = (n + blockSize - 1) / blockSize;
  intt_scale<<<gridSize, blockSize, 0, stream>>>(d_data, n, n_inverse, p);
  //cudaDeviceSynchronize();
}
