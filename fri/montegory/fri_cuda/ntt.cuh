#include <cuda_runtime.h>
#include "util.cuh"

#define PAD_FACTOR 4
#define PAD(x) (x + (x >> PAD_FACTOR))

// ── 입력 변환 커널: uint32_t 배열을 Montgomery domain으로 ────
__global__ void to_mont_kernel(uint32_t* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = to_mont(d_data[idx]);
}

// ── 출력 변환 커널: Montgomery domain에서 일반 값으로 ─────────
__global__ void from_mont_kernel(uint32_t* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = from_mont(d_data[idx]);
}

__global__ void bit_reverse_kernel(uint32_t* d_data, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int rev = 0, temp = idx;
    for (int i = 0; i < log_n; i++) { rev = (rev << 1) | (temp & 1); temp >>= 1; }
    if (idx < rev) {
        uint32_t t = d_data[idx]; d_data[idx] = d_data[rev]; d_data[rev] = t;
    }
}

// twiddle: Montgomery 형태로 저장돼 있음
// data:    Montgomery 형태로 처리
// → butterfly 안의 mul_mod = mont_mul → 결과도 Montgomery 형태 유지
__global__ void ntt_shared_kernel(uint32_t* d_data,
                                   const uint32_t* __restrict__ d_twiddles,
                                   uint32_t n) {
    __shared__ uint32_t s_data[SHARED_SIZE + (SHARED_SIZE >> PAD_FACTOR)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;

    if (block_offset + tid         < n) s_data[PAD(tid)]         = d_data[block_offset + tid];
    if (block_offset + tid + blockDim.x < n) s_data[PAD(tid + blockDim.x)] = d_data[block_offset + tid + blockDim.x];
    __syncthreads();

    for (int len = 2; len <= SHARED_SIZE; len <<= 1) {
        int pair_dist = len / 2;
        int group  = tid / pair_dist;
        int offset = tid % pair_dist;
        int i = group * len + offset;
        int j = i + pair_dist;

        uint32_t w  = d_twiddles[offset * (n / len)];
        uint32_t u  = s_data[PAD(i)];
        uint32_t v  = s_data[PAD(j)];
        uint32_t vw = mul_mod(v, w);

        s_data[PAD(i)] = add_mod(u, vw);
        s_data[PAD(j)] = sub_mod(u, vw);
        __syncthreads();
    }

    if (block_offset + tid         < n) d_data[block_offset + tid]         = s_data[PAD(tid)];
    if (block_offset + tid + blockDim.x < n) d_data[block_offset + tid + blockDim.x] = s_data[PAD(tid + blockDim.x)];
}

__global__ void ntt_global_kernel(uint32_t* d_data,
                                   const uint32_t* __restrict__ d_twiddles,
                                   int len, uint32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    int pair_dist = len / 2;
    int group  = tid / pair_dist;
    int offset = tid % pair_dist;
    int i = group * len + offset;
    int j = i + pair_dist;

    uint32_t w   = d_twiddles[offset * (n / len)];
    uint32_t u   = d_data[i];
    uint32_t v   = d_data[j];
    uint32_t vw  = mul_mod(v, w);

    d_data[i] = add_mod(u, vw);
    d_data[j] = sub_mod(u, vw);
}

// ntt(): 입력/출력은 일반 값, 내부는 Montgomery domain
// 경계: to_mont → [bit_reverse → butterfly] → from_mont
void ntt(uint32_t* d_data, uint32_t* d_twiddles, int n, cudaStream_t stream = 0) {
    int log_n = 0, temp = n;
    while (temp > 1) { log_n++; temp >>= 1; }

    int gridSize = (n + blockSize - 1) / blockSize;

    // 입력을 Montgomery domain으로 변환
    to_mont_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n);

    bit_reverse_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n, log_n);

    int shared_grid = n / SHARED_SIZE;
    ntt_shared_kernel<<<shared_grid, blockSize, 0, stream>>>(d_data, d_twiddles, n);

    for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        ntt_global_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, d_twiddles, len, n);
    }

    // 결과를 일반 값으로 변환
    gridSize = (n + blockSize - 1) / blockSize;
    from_mont_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n);
}
