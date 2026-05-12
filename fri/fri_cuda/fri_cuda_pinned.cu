#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <string>

// Poseidon12 상수 실체화 — 이 translation unit 하나에서만 선언
#define POSEIDON_DEFINE_CONSTANTS
#include "poseidon.cuh"

#define PAD_FACTOR 4 
#define PAD(x) (x + (x >> PAD_FACTOR))

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

// ===== Host Functions =====

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
    KernelParams host_params = {p, root, root_inv, root_pw, log_n, (int)n};
    cudaMemcpyToSymbol(c_params, &host_params, sizeof(KernelParams));
}

// simple_hash64 / sha256_hash 제거됨 — Week 2: GPU Poseidon으로 교체
// Merkle hashing 은 poseidon.cuh 의 poseidon_merkle_layer 커널 사용
// ===== Device Functions =====

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

// ===== Kernel Functions =====

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

__global__ void fft_shared_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles) {
    
    __shared__ uint64_t s_data[SHARED_SIZE + (SHARED_SIZE >> PAD_FACTOR)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;
    
    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    if (block_offset + idx1 < c_params.n)
        s_data[PAD(idx1)] = d_data[block_offset + idx1];
        
    if (block_offset + idx2 < c_params.n)
        s_data[PAD(idx2)] = d_data[block_offset + idx2];

    __syncthreads();

    for (int len = 2; len <= SHARED_SIZE; len <<= 1) {
        int pair_dist = len / 2;
        
        int group = tid / pair_dist;
        int offset = tid % pair_dist;
        
        int i = group * len + offset;
        int j = i + pair_dist;

        int twiddle_idx = offset * (c_params.n / len);
        uint64_t w = d_twiddles[twiddle_idx];

        uint64_t u = s_data[PAD(i)];
        uint64_t v = s_data[PAD(j)];
        
        uint64_t vw = mul_mod(v, w);

        s_data[PAD(i)] = add_mod(u, vw);
        s_data[PAD(j)] = sub_mod(u, vw);

        __syncthreads();
    }

    if (block_offset + idx1 < c_params.n)
        d_data[block_offset + idx1] = s_data[PAD(idx1)];
        
    if (block_offset + idx2 < c_params.n)
        d_data[block_offset + idx2] = s_data[PAD(idx2)];
}

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

__global__ void ifft_shared_kernel(
    uint64_t* d_data,
    const uint64_t* __restrict__ d_twiddles_inv) {
    
    __shared__ uint64_t s_data[SHARED_SIZE + (SHARED_SIZE >> PAD_FACTOR)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SHARED_SIZE;
    
    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    if (block_offset + idx1 < c_params.n)
        s_data[PAD(idx1)] = d_data[block_offset + idx1];
        
    if (block_offset + idx2 < c_params.n)
        s_data[PAD(idx2)] = d_data[block_offset + idx2];

    __syncthreads();

    for (int len = 2; len <= SHARED_SIZE; len <<= 1) {
        int pair_dist = len / 2;
        
        int group = tid / pair_dist;
        int offset = tid % pair_dist;
        
        int i = group * len + offset;
        int j = i + pair_dist;

        int twiddle_idx = offset * (c_params.n / len);
        uint64_t w_inv = d_twiddles_inv[twiddle_idx];

        uint64_t u = s_data[PAD(i)];
        uint64_t v = s_data[PAD(j)];
        uint64_t vw_inv = mul_mod(v, w_inv);

        s_data[PAD(i)] = add_mod(u, vw_inv);
        s_data[PAD(j)] = sub_mod(u, vw_inv);

        __syncthreads();
    }

    if (block_offset + idx1 < c_params.n)
        d_data[block_offset + idx1] = s_data[PAD(idx1)];
        
    if (block_offset + idx2 < c_params.n)
        d_data[block_offset + idx2] = s_data[PAD(idx2)];
}

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

__global__ void ifft_scale(uint64_t* d_data, int n, uint64_t n_inverse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    d_data[tid] = mul_mod(d_data[tid], n_inverse);
}

// merkle_layer_kernel 제거됨 — poseidon.cuh 의 poseidon_merkle_layer 사용

// ===== Host Helper Functions =====

void fft_optimized(uint64_t* d_data, uint64_t* d_twiddles, int n) {
    int gridSize = (n + blockSize - 1) / blockSize;
    int log_n = 0;
    int temp = n;
    while(temp > 1) { log_n++; temp >>= 1; }
    
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    int shared_grid_size = n / SHARED_SIZE;
    fft_shared_kernel<<<shared_grid_size, blockSize>>>(d_data, d_twiddles);
    cudaDeviceSynchronize();
    
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
    
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    
    int shared_grid_size = n / SHARED_SIZE;
    ifft_shared_kernel<<<shared_grid_size, blockSize>>>(d_data, d_twiddles_inv);
    cudaDeviceSynchronize();
    
    for (int len = SHARED_SIZE * 2; len <= n; len <<= 1) {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        ifft_global_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles_inv, len);
        cudaDeviceSynchronize();
    }
    
    uint64_t n_inverse = mod_inverse_prime(n, p);
    gridSize = (n + blockSize - 1) / blockSize;
    ifft_scale<<<gridSize, blockSize>>>(d_data, n, n_inverse);
    cudaDeviceSynchronize();
}

// ===== FRI Commitment Structure =====

struct FRICommitmentGPU {
    std::vector<std::string> layer_roots;
    std::vector<int> layer_sizes;
    std::vector<float> layer_times;
};

// ── Week 2: GPU Poseidon Merkle commitment ────────────────────────────────
// D→H memcpy + CPU hash 완전 제거.
// 모든 Merkle hashing 이 GPU 에서 실행됨.
// 핑퐁 버퍼 방식으로 race condition 없음.

FRICommitmentGPU fri_commitment_gpu(
    uint64_t* d_evals,
    uint64_t* /*d_twiddles_ifft — 미사용, 시그니처 호환 유지*/,
    int initial_size,
    int num_layers,
    uint64_t /*p*/) {

    FRICommitmentGPU result;

    // 핑퐁 버퍼 할당 (각각 initial_size 크기)
    uint64_t* d_buf[2];
    cudaMalloc(&d_buf[0], initial_size * sizeof(uint64_t));
    cudaMalloc(&d_buf[1], initial_size * sizeof(uint64_t));

    // 입력을 buf[0] 에 복사
    cudaMemcpy(d_buf[0], d_evals, initial_size * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    int cur_size = initial_size;
    int src = 0, dst = 1;

    std::cout << "\nFRI Commitment Layers (GPU Poseidon12):" << std::endl;

    for (int layer = 0; layer < num_layers && cur_size > 1; layer++) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);

        // ── GPU Poseidon Merkle 레이어 ──────────────────────────────────
        // 현재 cur_size 개 노드 → cur_size/2 개 부모 노드
        int n_pairs = cur_size / 2;
        int grid    = (n_pairs + blockSize - 1) / blockSize;
        poseidon_merkle_layer<<<grid, blockSize>>>(d_buf[src], d_buf[dst], n_pairs);
        cudaDeviceSynchronize();

        cudaEventRecord(t1);
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

        // 핑퐁
        std::swap(src, dst);
        cur_size = n_pairs;

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
    return result;
}

// ===== Main =====

int main(int argc, char** argv) {
    std::cout << "=== CUDA FFT/IFFT with FRI Commitment ===" << std::endl << std::endl;
    
    int log_n = std::stoi(argv[1]);
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;

    // Poseidon12 상수를 constant memory 에 로드
    cudaError_t poseidon_err = poseidon_init();
    if (poseidon_err != cudaSuccess) {
        std::cerr << "poseidon_init failed: "
                  << cudaGetErrorString(poseidon_err) << std::endl;
        return 1;
    }

    {
    	uint64_t h_in[2]  = {0ULL, 0ULL};
    	uint64_t h_out    = 0;
    	uint64_t *d_in, *d_out;
    	cudaMalloc(&d_in,  2*sizeof(uint64_t));
    	cudaMalloc(&d_out,   sizeof(uint64_t));
    	cudaMemcpy(d_in, h_in, 2*sizeof(uint64_t), cudaMemcpyHostToDevice);
    	poseidon_merkle_layer<<<1,1>>>(d_in, d_out, 1);
    	cudaDeviceSynchronize();
    	cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    	std::cout << "Poseidon bit-exact: compress(0,0) = " << h_out
              << (h_out == 877077992ULL ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
        cudaFree(d_in); cudaFree(d_out);
    }
    std::cout << "Poseidon12 constants loaded." << std::endl;

    set_constants(p, root, 1ULL << 20, log_n, n);
    
    size_t bytes = n * sizeof(uint64_t);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Polynomial size: 2^" << log_n << " = " << n << std::endl;
    std::cout << "  Field modulus: " << p << std::endl;
    std::cout << "  Memory required: " << (bytes / (1024*1024)) << " MB" << std::endl << std::endl;
    
    // Host memory
    uint64_t* h_coeff = (uint64_t*)malloc(bytes);
    for(uint64_t i = 0; i < n; i++) {
        h_coeff[i] = i % p;
    }
    
    // Device memory
    uint64_t *d_coeff, *d_twiddles_fft, *d_twiddles_ifft, *d_evals;
    cudaMallocHost(&d_coeff, bytes);
    cudaMallocHost(&d_twiddles_fft, bytes);
    cudaMallocHost(&d_twiddles_ifft, bytes);
    cudaMallocHost(&d_evals, bytes);
    
    // Generate twiddle factors
    uint64_t generator = get_generator(root, p - 1, n, p);
    uint64_t generator_inv = mod_inverse_prime(generator, p);
    
    std::cout << "Generating twiddle factors..." << std::endl;
    int gridSize = (n + blockSize - 1) / blockSize;
    generate<<<gridSize, blockSize>>>(d_twiddles_fft, d_twiddles_ifft, generator, generator_inv, n);
    cudaDeviceSynchronize();
    
    // Transfer data to device
    cudaMemcpy(d_coeff, h_coeff, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_evals, h_coeff, bytes, cudaMemcpyHostToDevice);
    
    // Timing events
    cudaEvent_t total_start, total_stop, fft_start, fft_stop, fri_start, fri_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&fft_start);
    cudaEventCreate(&fft_stop);
    cudaEventCreate(&fri_start);
    cudaEventCreate(&fri_stop);
    
    cudaEventRecord(total_start);
    
    // ===== FFT =====
    std::cout << "\n--- Forward FFT ---" << std::endl;
    cudaEventRecord(fft_start);
    
    fft_optimized(d_coeff, d_twiddles_fft, n);
    
    cudaEventRecord(fft_stop);
    cudaEventSynchronize(fft_stop);
    float fft_ms = 0;
    cudaEventElapsedTime(&fft_ms, fft_start, fft_stop);
    std::cout << "FFT time: " << std::fixed << std::setprecision(2) << fft_ms << " ms" << std::endl;
    
    // Copy evaluations
    cudaMemcpy(d_evals, d_coeff, bytes, cudaMemcpyDeviceToDevice);
    
    // ===== IFFT =====
    std::cout << "\n--- Inverse FFT ---" << std::endl;
    cudaEventRecord(fft_start);
    
    ifft_optimized(d_coeff, d_twiddles_ifft, n, p);
    
    cudaEventRecord(fft_stop);
    cudaEventSynchronize(fft_stop);
    float ifft_ms = 0;
    cudaEventElapsedTime(&ifft_ms, fft_start, fft_stop);
    std::cout << "IFFT time: " << std::fixed << std::setprecision(2) << ifft_ms << " ms" << std::endl;
    
    // ===== FRI Commitment =====
    std::cout << "\n--- FRI Commitment ---" << std::endl;
    cudaEventRecord(fri_start);
    
    FRICommitmentGPU fri = fri_commitment_gpu(d_evals, d_twiddles_ifft, n, log_n, p);
    
    cudaEventRecord(fri_stop);
    cudaEventSynchronize(fri_stop);
    float fri_ms = 0;
    cudaEventElapsedTime(&fri_ms, fri_start, fri_stop);
    std::cout << "FRI commitment time: " << std::fixed << std::setprecision(2) << fri_ms << " ms" << std::endl;
    
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    
    // ===== Verification =====
    std::cout << "\n--- Correctness Verification ---" << std::endl;
    
    uint64_t* h_result = (uint64_t*)malloc(bytes);
    cudaMemcpy(h_result, d_coeff, bytes, cudaMemcpyDeviceToHost);
    
    bool all_match = true;
    int error_count = 0;
    for(uint64_t i = 0; i < n; i++) {
        uint64_t expected = i % p;
        if(h_result[i] != expected) {
            if(error_count < 5) {
                std::cout << "Error at " << i << ": expected " << expected 
                          << ", got " << h_result[i] << std::endl;
            }
            error_count++;
            all_match = false;
        }
    }
    
    if(all_match) {
        std::cout << "✓ FFT → IFFT verification passed! All values match." << std::endl;
    } else {
        std::cout << "✗ Verification failed: " << error_count << " mismatches" << std::endl;
    }
    
    // ===== Performance Summary =====
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::left << std::setw(25) << "FFT (forward):" 
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
              << fft_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "IFFT (inverse):" 
              << std::right << std::setw(10) << ifft_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "FRI commitment:" 
              << std::right << std::setw(10) << fri_ms << " ms" << std::endl;
    std::cout << std::left << std::setw(25) << "Total:" 
              << std::right << std::setw(10) << total_ms << " ms" << std::endl;
    
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  FFT: " << std::fixed << std::setprecision(2) 
              << (n * log_n / 1e6 / fft_ms) << " GOP/s" << std::endl;
    std::cout << "  IFFT: " << (n * log_n / 1e6 / ifft_ms) << " GOP/s" << std::endl;
    
    // Cleanup
    free(h_coeff);
    free(h_result);
    cudaFree(d_coeff);
    cudaFree(d_evals);
    cudaFree(d_twiddles_fft);
    cudaFree(d_twiddles_ifft);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(fft_start);
    cudaEventDestroy(fft_stop);
    cudaEventDestroy(fri_start);
    cudaEventDestroy(fri_stop);
    
    return 0;
}

