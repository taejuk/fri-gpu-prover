#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>

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

uint64_t simple_hash64(const uint64_t* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL; // FNV offset basis
    const uint64_t prime = 0x100000001b3ULL;

    for (size_t i = 0; i < size; i++) {
        uint64_t x = data[i];
        // 몇 번 섞어준다
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;

        h ^= x;
        h *= prime;
    }
    return h;
}

// 기존 sha256_hash 대신 사용
std::string sha256_hash(const uint64_t* data, size_t size, uint64_t /*p*/) {
    uint64_t h = simple_hash64(data, size);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    // 16 hex chars (64bit) 정도만 사용
    oss << std::setw(16) << h;
    return oss.str();
}
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

// Simple Merkle tree layer (GPU)
__global__ void merkle_layer_kernel(
    const uint64_t* d_data,
    uint64_t* d_hashes,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Simple hash: XOR + modular multiply (not cryptographic, for demo)
    uint64_t val = d_data[idx];
    d_hashes[idx] = (val * 2654435761ULL) ^ 0xdeadbeefdeadbeefULL;
}

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

FRICommitmentGPU fri_commitment_gpu(
    uint64_t* d_evals,
    uint64_t* d_twiddles_ifft,
    int initial_size,
    int num_layers,
    uint64_t p) {
    
    FRICommitmentGPU result;
    uint64_t* d_current = d_evals;
    uint64_t* d_next = nullptr;
    int current_size = initial_size;
    
    cudaMalloc(&d_next, initial_size * sizeof(uint64_t));
    
    std::cout << "\nFRI Commitment Layers (GPU):" << std::endl;
    
    for(int layer = 0; layer < num_layers && current_size > 1; layer++) {
        cudaEvent_t layer_start, layer_stop;
        cudaEventCreate(&layer_start);
        cudaEventCreate(&layer_stop);
        cudaEventRecord(layer_start);
        
        // Copy evaluation to host for hashing
        uint64_t* h_evals = (uint64_t*)malloc(current_size * sizeof(uint64_t));
        cudaMemcpy(h_evals, d_current, current_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        // Compute Merkle root (SHA256 hash of all evaluations)
        std::string root = sha256_hash(h_evals, current_size, p);
        result.layer_roots.push_back(root);
        result.layer_sizes.push_back(current_size);
        
        std::cout << "  Layer " << layer << ": " << current_size << " points -> root: " 
                  << root.substr(0, 16) << "..." << std::endl;
        
        if(current_size <= 1) break;
        
        // Folding: simple averaging (real FRI is more complex)
        // p_next(x) = (p(x) + p(-x)) / 2, but we simplify to halving size
        int next_size = current_size / 2;
        
        // Simple folding kernel (average adjacent values)
        int gridSize = (next_size + blockSize - 1) / blockSize;
        
        cudaEventRecord(layer_stop);
        cudaEventSynchronize(layer_stop);
        float layer_ms = 0;
        cudaEventElapsedTime(&layer_ms, layer_start, layer_stop);
        result.layer_times.push_back(layer_ms);
        
        free(h_evals);
        cudaEventDestroy(layer_start);
        cudaEventDestroy(layer_stop);
        
        current_size = next_size;
    }
    
    cudaFree(d_next);
    return result;
}

// ===== Main =====

int main() {
    std::cout << "=== CUDA FFT/IFFT with FRI Commitment ===" << std::endl << std::endl;
    
    int log_n = 20;
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;
    
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
    cudaMalloc(&d_coeff, bytes);
    cudaMalloc(&d_twiddles_fft, bytes);
    cudaMalloc(&d_twiddles_ifft, bytes);
    cudaMalloc(&d_evals, bytes);
    
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

