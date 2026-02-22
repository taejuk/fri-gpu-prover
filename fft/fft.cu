#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

struct KernelParams {
    uint64_t p;
    uint64_t root;
    uint64_t root_inv;
    uint64_t root_pw;
    int log_n;
    int n;
};

const int blockSize = 256;

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

// 생성: g^i 계산
__global__ void generate(uint64_t* d_data,uint64_t* d_data_inv, uint64_t g,uint64_t g_inv ,int size) {
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



// Bit-reversal permutation
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

// FFT Butterfly (정방향)
__global__ void fft_butterfly_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles, int len) {
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

// IFFT Butterfly (역방향 twiddles 사용)
__global__ void ifft_butterfly_kernel(uint64_t* d_data, const uint64_t* __restrict__ d_twiddles_inv, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = c_params.n;
    
    if (tid >= n / 2) return;

    int pair_dist = len / 2;
    int group = tid / pair_dist;      
    int offset = tid % pair_dist;     

    int i = group * len + offset;     
    int j = i + pair_dist;            

    int twiddle_idx = offset * (n / len);
    uint64_t w_inv = d_twiddles_inv[twiddle_idx];  // 역 twiddle

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




void fft(uint64_t* d_data, uint64_t* d_twiddles, int n) {
    // Bit-reversal 먼저
    int gridSize = (n + blockSize - 1) / blockSize;
    int log_n = 0;
    int temp = n;
    while(temp > 1) { log_n++; temp >>= 1; }
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    for(int len = 2; len <= n; len *= 2) {
        gridSize = (n/2 + blockSize - 1) / blockSize;
        fft_butterfly_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles, len);
        cudaDeviceSynchronize();
    }
}

void ifft(uint64_t* d_data, uint64_t* d_twiddles_inv, int n, uint64_t p) {
    int gridSize = (n + blockSize - 1) / blockSize;
    int log_n = 0;
    int temp = n;
    while(temp > 1) { log_n++; temp >>= 1; }
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    // IFFT 스테이지 (역 twiddles 사용)
    for(int len = 2; len <= n; len *= 2) {
        gridSize = (n/2 + blockSize - 1) / blockSize;
        ifft_butterfly_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles_inv, len);
        cudaDeviceSynchronize();
    }
    
    // 스케일링: 1/n
    uint64_t n_inverse = mod_inverse_prime(n, p);
    gridSize = (n + blockSize - 1) / blockSize;
    ifft_scale<<<gridSize, blockSize>>>(d_data, n, n_inverse);
    cudaDeviceSynchronize();
}

// ===== Main =====

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    int log_n = 20;
    uint64_t p = 2013265921ULL;  // BabyBear
    uint64_t root = 7;
    uint64_t root_pw = 1ULL << 20;
    uint64_t n = 1ULL << 20;
    set_constants(p, root, root_pw, log_n, n);
    
    int domainSize = 1 << log_n;
    size_t bytes = domainSize * sizeof(uint64_t);

    // ===== 호스트 메모리 =====
    uint64_t* h_domain = (uint64_t*)malloc(bytes);
    uint64_t* h_coeff = (uint64_t*)malloc(bytes);
    uint64_t* h_twiddles_fft = (uint64_t*)malloc(bytes);
    uint64_t* h_twiddles_ifft = (uint64_t*)malloc(bytes);

    // 계수 초기화
    for(int i = 0; i < domainSize; i++) {
        h_coeff[i] = i % p;
    }

    // ===== 디바이스 메모리 =====
    uint64_t* d_domain, *d_coeff, *d_twiddles_fft, *d_twiddles_ifft;
    cudaMalloc(&d_domain, bytes);
    cudaMalloc(&d_coeff, bytes);
    cudaMalloc(&d_twiddles_fft, bytes);
    cudaMalloc(&d_twiddles_ifft, bytes);

    // ===== Twiddle factors 계산 =====
    uint64_t generator = get_generator(root, p - 1, domainSize, p);
    uint64_t generator_inv = mod_inverse_prime(generator, p);
    std::cout << "Generator: " << generator << std::endl;
    
    
    int gridSize = (n + blockSize -1) / blockSize;
    generate<<<gridSize, blockSize>>>(d_twiddles_fft,d_twiddles_ifft, generator,generator_inv,n);
    // uint64_t w = 1;
    // for(int i = 0; i < domainSize; i++) {
    //     h_twiddles_fft[i] = w;
    //     w = multiply_uint64(w, generator, p);
    // }
    
    
    // w = 1;
    // for(int i = 0; i < domainSize; i++) {
    //     h_twiddles_ifft[i] = w;
    //     w = multiply_uint64(w, generator_inv, p);
    // }
    

    cudaMemcpy(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice);
    
    
    // ===== FFT 실행 =====
    std::cout << "FFT 시작..." << std::endl;
    fft(d_coeff, d_twiddles_fft, domainSize);
    
    // 결과 저장
    uint64_t* h_coeff_after_fft = (uint64_t*)malloc(bytes);
    cudaMemcpy(h_coeff_after_fft, d_coeff, bytes, cudaMemcpyDeviceToHost);

    // ===== IFFT 실행 =====
    std::cout << "IFFT 시작..." << std::endl;
    ifft(d_coeff, d_twiddles_ifft, domainSize, p);

    // ===== 결과 검증 =====
    cudaMemcpy(h_coeff, d_coeff, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
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

    // ===== 메모리 해제 =====
    free(h_domain);
    free(h_coeff);
    free(h_twiddles_fft);
    free(h_twiddles_ifft);
    free(h_coeff_after_fft);
    
    cudaFree(d_domain);
    cudaFree(d_coeff);
    cudaFree(d_twiddles_fft);
    cudaFree(d_twiddles_ifft);

    return 0;
}

