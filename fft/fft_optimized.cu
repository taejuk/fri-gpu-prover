#include <iostream>
#include <vector>
#include <cuda_runtime.h>

struct KernelParams {
    uint64_t p;
    uint64_t root;
    uint64_t root_inv;
    uint64_t root_pw;
    int log_n;
    int n;
};

const int blockSize = 512;
const int SHARED_SIZE = 1024;
const int SHARED_STAGES = 10;

__constant__ KernelParams c_params;

// ===== 호스트 함수 =====
uint64_t multiply_uint64(uint64_t a, uint64_t b, uint64_t p) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    return (uint64_t)(mul % p);
}

uint64_t mod_inverse_prime(uint64_t a, uint64_t p) {
    uint64_t result = 1;
    uint64_t base = a;
    uint64_t exp = p - 2;
    
    while(exp > 0) {
        if(exp & 1) {
            result = ((__uint128_t)result * base) % p;
        }
        base = ((__uint128_t)base * base) % p;
        exp >>= 1;
    }
    return result;
}

uint64_t get_generator(uint64_t g, uint64_t n, uint64_t m, uint64_t p) {
    uint64_t pow = n / m;
    uint64_t res = 1;
    uint64_t base = g;
    
    while(pow > 0) {
        if(pow & 1) {
            res = multiply_uint64(res, base, p);
        }
        base = multiply_uint64(base, base, p);
        pow = pow >> 1;
    }
    return res;
}

void set_constants(uint64_t p, uint64_t root, uint64_t root_pw, int log_n, uint64_t n) {
    uint64_t root_inv = mod_inverse_prime(root, p);
    KernelParams host_params = {p, root, root_inv, root_pw, log_n, n};
    cudaMemcpyToSymbol(c_params, &host_params, sizeof(KernelParams));
}

// ===== 디바이스 함수 =====
__device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    return (uint64_t)(mul % c_params.p);
}

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t res = a + b;
    if (res >= c_params.p) res -= c_params.p;
    return res;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a - b + c_params.p);
}

// ===== 커널 함수 =====

// Twiddle 생성 커널
__global__ void generate(uint64_t* d_data, uint64_t* d_data_inv, uint64_t g, uint64_t g_inv, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        uint64_t res = 1;
        uint64_t res_inv = 1;
        uint64_t base = g;
        uint64_t base_inv = g_inv;
        int exp = idx;
    
        while(exp > 0) {
            if(exp & 1) {
                res = mul_mod(res, base);
                res_inv = mul_mod(res_inv, base_inv);
            }
            base = mul_mod(base, base);
            base_inv = mul_mod(base_inv, base_inv);
            exp = exp >> 1;
        }
        d_data[idx] = res;
        d_data_inv[idx] = res_inv;
    }
}

// Bit-reversal
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

// ★ Shared Memory를 이용한 FFT (처음 10단계)
__global__ void fft_shared_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles) {
    
    // 공유 메모리: 1024개 요소
    __shared__ uint64_t s_data[SHARED_SIZE];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;
    
    // ===== Phase 1: Global → Shared (Coalesced Load) =====
    // 512 스레드가 1024개 데이터를 로드 (각 스레드가 2개씩)
    s_data[tid] = d_data[block_offset + tid];
    s_data[tid + blockSize] = d_data[block_offset + tid + blockSize];
    __syncthreads();

    // ===== Phase 2: Shared Memory에서 10단계 FFT =====
    for (int stage = 0; stage < SHARED_STAGES; stage++) {
        int len = 1 << (stage + 1);  // 2, 4, 8, ..., 1024
        int pair_dist = len / 2;
        
        // 각 스레드가 처리할 버터플라이 인덱스 계산
        int butterfly_group = tid / pair_dist;
        int butterfly_offset = tid % pair_dist;
        
        int i = butterfly_group * len + butterfly_offset;
        int j = i + pair_dist;

        // Twiddle factor (전체 N 기준)
        int n = c_params.n;
        int twiddle_idx = butterfly_offset * (n / len);
        uint64_t w = d_twiddles[twiddle_idx];

        uint64_t u = s_data[i];
        uint64_t v = s_data[j];
        uint64_t vw = mul_mod(v, w);

        s_data[i] = add_mod(u, vw);
        s_data[j] = sub_mod(u, vw);

        __syncthreads();  // 모든 스레드가 동기화
    }

    // ===== Phase 3: Shared → Global (Coalesced Store) =====
    d_data[block_offset + tid] = s_data[tid];
    d_data[block_offset + tid + blockSize] = s_data[tid + blockSize];
}

// ★ Global Memory를 이용한 FFT (나머지 10단계: len >= 2048)
__global__ void fft_global_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles,
    int len) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = c_params.n;
    
    if (tid >= n / 2) return;

    int pair_dist = len / 2;
    int group = tid / pair_dist;      
    int offset = tid % pair_dist;     

    int i = group * len + offset;     
    int j = i + pair_dist;            

    int twiddle_idx = offset * (n / len);
    uint64_t w = d_twiddles[twiddle_idx];

    uint64_t u = d_data[i];
    uint64_t v = d_data[j];
    uint64_t vw = mul_mod(v, w);

    d_data[i] = add_mod(u, vw);
    d_data[j] = sub_mod(u, vw);
}

// IFFT Butterfly (Shared Memory)
__global__ void ifft_shared_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles_inv) {
    
    __shared__ uint64_t s_data[SHARED_SIZE];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;
    
    // Load
    s_data[tid] = d_data[block_offset + tid];
    s_data[tid + blockSize] = d_data[block_offset + tid + blockSize];
    __syncthreads();

    // 10단계 IFFT
    for (int stage = 0; stage < SHARED_STAGES; stage++) {
        int len = 1 << (stage + 1);
        int pair_dist = len / 2;
        
        int butterfly_group = tid / pair_dist;
        int butterfly_offset = tid % pair_dist;
        
        int i = butterfly_group * len + butterfly_offset;
        int j = i + pair_dist;

        int n = c_params.n;
        int twiddle_idx = butterfly_offset * (n / len);
        uint64_t w_inv = d_twiddles_inv[twiddle_idx];

        uint64_t u = s_data[i];
        uint64_t v = s_data[j];
        uint64_t vw_inv = mul_mod(v, w_inv);

        s_data[i] = add_mod(u, vw_inv);
        s_data[j] = sub_mod(u, vw_inv);

        __syncthreads();
    }

    // Store
    d_data[block_offset + tid] = s_data[tid];
    d_data[block_offset + tid + blockSize] = s_data[tid + blockSize];
}

// IFFT Global
__global__ void ifft_global_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles_inv,
    int len) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = c_params.n;
    
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
    uint64_t vw_inv = mul_mod(v, w_inv);

    d_data[i] = add_mod(u, vw_inv);
    d_data[j] = sub_mod(u, vw_inv);
}

// IFFT 스케일링
__global__ void ifft_scale(uint64_t* d_data, int n, uint64_t n_inverse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    d_data[tid] = mul_mod(d_data[tid], n_inverse);
}

// ===== 호스트 레벨 함수 =====

void fft_optimized(uint64_t* d_data, uint64_t* d_twiddles, int n) {
    int gridSize = (n + blockSize - 1) / blockSize;
    int log_n = 0;
    int temp = n;
    while(temp > 1) { log_n++; temp >>= 1; }
    
    // Bit-reversal
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    // ★ Shared memory FFT (처음 10단계: len = 2, 4, ..., 1024)
    int shared_grid_size = n / SHARED_SIZE;  // 1024 블록
    fft_shared_kernel<<<shared_grid_size, blockSize>>>(d_data, d_twiddles);
    cudaDeviceSynchronize();
    
    // ★ Global memory FFT (나머지 10단계: len = 2048, 4096, ..., 2^20)
    for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        fft_global_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles, len);
        cudaDeviceSynchronize();
    }
}

void ifft_optimized(uint64_t* d_data, uint64_t* d_twiddles_inv, int n, uint64_t p) {
    int gridSize = (n + blockSize - 1) / blockSize;
    int log_n = 0;
    int temp = n;
    while(temp > 1) { log_n++; temp >>= 1; }
    
    // Bit-reversal
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    // ★ Shared memory IFFT (처음 10단계)
    int shared_grid_size = n / SHARED_SIZE;
    ifft_shared_kernel<<<shared_grid_size, blockSize>>>(d_data, d_twiddles_inv);
    cudaDeviceSynchronize();
    
    // ★ Global memory IFFT (나머지 10단계)
    for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        ifft_global_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles_inv, len);
        cudaDeviceSynchronize();
    }
    
    // 스케일링
    uint64_t n_inverse = mod_inverse_prime(n, p);
    gridSize = (n + blockSize - 1) / blockSize;
    ifft_scale<<<gridSize, blockSize>>>(d_data, n, n_inverse);
    cudaDeviceSynchronize();
}

// ===== Main =====

int main() {
    int log_n = 20;
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t root_pw = 1ULL << 20;
    uint64_t n = 1ULL << 20;
    set_constants(p, root, root_pw, log_n, n);
    
    int domainSize = 1 << log_n;
    size_t bytes = domainSize * sizeof(uint64_t);

    // 호스트 메모리
    uint64_t* h_coeff = (uint64_t*)malloc(bytes);
    for(int i = 0; i < domainSize; i++) {
        h_coeff[i] = i % p;
    }

    // 디바이스 메모리
    uint64_t *d_coeff, *d_twiddles_fft, *d_twiddles_ifft;
    cudaMalloc(&d_coeff, bytes);
    cudaMalloc(&d_twiddles_fft, bytes);
    cudaMalloc(&d_twiddles_ifft, bytes);

    // Twiddle 계산
    uint64_t generator = get_generator(root, p - 1, domainSize, p);
    uint64_t generator_inv = mod_inverse_prime(generator, p);
    std::cout << "Generator: " << generator << std::endl;
    
    int gridSize = (n + blockSize - 1) / blockSize;
    generate<<<gridSize, blockSize>>>(d_twiddles_fft, d_twiddles_ifft, generator, generator_inv, n);
    cudaDeviceSynchronize();

    // 데이터 전송
    cudaMemcpy(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice);

    // 타이밍
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fft_optimized(d_coeff, d_twiddles_fft, domainSize);
    
    ifft_optimized(d_coeff, d_twiddles_ifft, domainSize, p);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "\n최적화된 FFT/IFFT 실행 시간: " << ms << " ms" << std::endl;

    // 결과 검증
    cudaMemcpy(h_coeff, d_coeff, bytes, cudaMemcpyDeviceToHost);

    std::cout << "\n검증 결과:" << std::endl;
    bool all_match = true;
    int error_count = 0;
    for(int i = 0; i < domainSize; i++) {
        uint64_t expected = i % p;
        if(h_coeff[i] != expected) {
            if(error_count < 10) {
                std::cout << "Error at index " << i 
                          << ": expected " << expected 
                          << ", got " << h_coeff[i] << std::endl;
            }
            error_count++;
            all_match = false;
        }
    }

    if(all_match) {
        std::cout << "FFT → IFFT 검증 성공! 모든 값이 일치합니다." << std::endl;
    } else {
        std::cout << "오류: " << error_count << "개 위치에서 불일치" << std::endl;
    }

    free(h_coeff);
    cudaFree(d_coeff);
    cudaFree(d_twiddles_fft);
    cudaFree(d_twiddles_ifft);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

