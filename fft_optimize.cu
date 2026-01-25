#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}
// p = 2^64 - 2^32 + 1
const uint64_t P = 0xFFFFFFFF00000001ULL;

struct KernelParams {
    uint64_t p;
    int n;
};

__constant__ KernelParams c_params;


__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t res = a + b;
    if (res < a || res >= c_params.p) res -= c_params.p;
    return res;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a - b + c_params.p);
}

__device__ __forceinline__ uint64_t multiply_uint64(uint64_t a, uint64_t b) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)mul;
    uint64_t hi = (uint64_t)(mul >> 64);
    
    uint64_t hi_shifted = hi << 32;
    uint64_t res = lo - hi + hi_shifted;

    const uint64_t EPSILON = 0xFFFFFFFF;
    if (lo < hi) res -= EPSILON;
    if (res >= c_params.p) res -= c_params.p;

    return res;
}

// 거듭제곱 (Twiddle 생성용)
__device__ uint64_t pow_mod(uint64_t base, uint64_t exp) {
    uint64_t res = 1;
    while (exp > 0) {
        if (exp & 1) res = multiply_uint64(res, base);
        base = multiply_uint64(base, base);
        exp >>= 1;
    }
    return res;
}

__global__ void init_twiddles_kernel(uint64_t* d_twiddles, uint64_t root_n, int n_half) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;
    d_twiddles[idx] = pow_mod(root_n, (uint64_t)idx);
}

// [2] Bit Reversal (입력 데이터 섞기)
__global__ void bit_reverse_kernel(uint64_t* d_data, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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

__global__ void fft_butterfly_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles, int len, int stride) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = c_params.n;

    if (tid >= n / 2) return;

    
    int half_len = len >> 1;
    int group = tid / half_len;      
    int offset = tid % half_len;     

    int i = group * len + offset;
    int j = i + half_len;

    
    int twiddle_idx = offset * stride;
    uint64_t w = d_twiddles[twiddle_idx];

    // 연산
    uint64_t u = d_data[i];
    uint64_t v = d_data[j];
    
    uint64_t vw = multiply_uint64(v, w); // v * w

    d_data[i] = add_mod(u, vw);          // u + vw
    d_data[j] = sub_mod(u, vw);          // u - vw
}


uint64_t pow_mod_host(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    unsigned __int128 b = base;
    while (exp > 0) {
        if (exp & 1) res = (uint64_t)((unsigned __int128)res * b % mod);
        b = b * b % mod;
        exp >>= 1;
    }
    return res;
}

int main() {
    
    int log_n = 20; 
    int N = 1 << log_n;
    
    uint64_t root_pw_gen = 1753635133440165772ULL; 
    
    uint64_t root_n = pow_mod_host(root_pw_gen, (1ULL << 32) / N, P);

    std::cout << "FFT Setup: N=2^" << log_n << " (" << N << ")" << std::endl;
    std::cout << "Root_N: " << root_n << std::endl;

    KernelParams h_params = {P, N};
    CHECK_CUDA(cudaMemcpyToSymbol(c_params, &h_params, sizeof(KernelParams)));

    
    size_t data_bytes = N * sizeof(uint64_t);
    size_t twiddle_bytes = (N / 2) * sizeof(uint64_t);
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t *h_data, *d_data, *d_twiddles;
    CHECK_CUDA(cudaMallocHost((void**)&h_data, data_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_twiddles, twiddle_bytes));

    // 입력 데이터 (테스트용: 0, 1, 2, ... N-1)
    for(int i=0; i<N; i++) h_data[i] = i; 
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    
    int blockSize = 256;
    
    
    int gridTwiddle = (N / 2 + blockSize - 1) / blockSize;
    init_twiddles_kernel<<<gridTwiddle, blockSize>>>(d_twiddles, root_n, N / 2);
    CHECK_CUDA(cudaGetLastError());

    int gridFull = (N + blockSize - 1) / blockSize;
    bit_reverse_kernel<<<gridFull, blockSize>>>(d_data, N, log_n);
    CHECK_CUDA(cudaGetLastError());

    int num_butterflies = N / 2;
    int gridButterfly = (num_butterflies + blockSize - 1) / blockSize;

    std::cout << "Running FFT..." << std::endl;
    for (int len = 2; len <= N; len <<= 1) {
        int stride = N / len;
        fft_butterfly_kernel<<<gridButterfly, blockSize>>>(d_data, d_twiddles, len, stride);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    
    CHECK_CUDA(cudaMemcpy(h_data, d_data, data_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "duration: " << duration.count() << " ms" << std::endl;
    
    cudaFree(d_data); cudaFree(d_twiddles); cudaFreeHost(h_data);
    return 0;
}
