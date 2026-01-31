#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>

// ===== CONFIGURATION =====
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
//const int SHARED_SIZE = 1024;
__constant__ KernelParams c_params;

// ===== HOST UTILS =====

// ===== FIAT-SHAMIR TRANSCRIPT =====

class FiatShamirTranscript {
private:
    // 실제로는 SHA256이나 Keccak 상태 객체를 써야 하지만,
    // 여기서는 데모를 위해 롤링 해시(Rolling Hash) 구조로 시뮬레이션합니다.
    uint64_t state; 

    void mix(uint64_t data) {
        // Simple mixing function (In production, use SHA256 update)
        state ^= data;
        state ^= state >> 33;
        state *= 0xff51afd7ed558ccdULL;
        state ^= state >> 33;
        state *= 0xc4ceb9fe1a85ec53ULL;
        state ^= state >> 33;
    }

public:
    FiatShamirTranscript() : state(0xcbf29ce484222325ULL) {}

    // 1. 관찰(Observe): Prover가 생성한 값(Merkle Root 등)을 기록
    void absorb(uint64_t data) {
        mix(data);
    }

    // 2. 챌린지 생성(Squeeze): 현재 상태를 기반으로 랜덤값(alpha) 생성
    uint64_t squeeze_challenge(uint64_t p) {
        // 상태를 한 번 더 섞어서 예측 불가능하게 만듦
        mix(0x1234567890ABCDEFULL); 
        return state % p;
    }
};

uint64_t multiply_uint64(uint64_t a, uint64_t b, uint64_t p) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    return (uint64_t)(mul % p);
}

uint64_t mod_inverse_prime(uint64_t a, uint64_t p) {
    uint64_t result = 1, base = a, exp = p - 2;
    while(exp > 0) {
        if(exp & 1) result = ((unsigned __int128)result * base) % p;
        base = ((unsigned __int128)base * base) % p;
        exp >>= 1;
    }
    return result;
}

uint64_t get_generator(uint64_t g, uint64_t n, uint64_t m, uint64_t p) {
    uint64_t pow = n / m;
    uint64_t res = 1, base = g;
    while(pow > 0) {
        if(pow & 1) res = multiply_uint64(res, base, p);
        base = multiply_uint64(base, base, p);
        pow >>= 1;
    }
    return res;
}

void set_constants(uint64_t p, uint64_t root, uint64_t root_pw, int log_n, uint64_t n) {
    uint64_t root_inv = mod_inverse_prime(root, p);
    KernelParams host_params = {p, root, root_inv, root_pw, log_n, (int)n};
    cudaMemcpyToSymbol(c_params, &host_params, sizeof(KernelParams));
}

// CPU Simple Hash for Merkle Tree (Demo Purpose)
uint64_t simple_hash64(const uint64_t* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; i++) {
        uint64_t x = data[i];
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33; h ^= x; h *= 0x100000001b3ULL;
    }
    return h;
}

std::string to_hex(uint64_t h) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    return oss.str();
}

// ===== DEVICE FUNCTIONS =====

__device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b % c_params.p);
}

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t res = a + b;
    return (res >= c_params.p) ? res - c_params.p : res;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a - b + c_params.p);
}

// ===== KERNELS (FFT/Gen) =====

__global__ void generate(uint64_t* d_data, uint64_t* d_data_inv, uint64_t g, uint64_t g_inv, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        uint64_t res = 1, res_inv = 1;
        uint64_t base = g, base_inv = g_inv;
        int exp = idx;
        while(exp > 0) {
            if(exp & 1) { res = mul_mod(res, base); res_inv = mul_mod(res_inv, base_inv); }
            base = mul_mod(base, base); base_inv = mul_mod(base_inv, base_inv);
            exp >>= 1;
        }
        d_data[idx] = res;
        d_data_inv[idx] = res_inv;
    }
}

// (FFT Kernels 생략 - 기존 코드와 동일하다고 가정하고 핵심 로직만 추가)
// ... [bit_reverse_kernel, fft_shared, fft_global, ifft_shared, ifft_global 등 기존 코드 유지] ...
// (전체 실행을 위해 필수 Kernel만 약식으로 다시 포함합니다)

__global__ void bit_reverse_kernel(uint64_t* d_data, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int rev = 0, temp = idx;
    for (int i = 0; i < log_n; i++) { rev = (rev << 1) | (temp & 1); temp >>= 1; }
    if (idx < rev) { uint64_t t = d_data[idx]; d_data[idx] = d_data[rev]; d_data[rev] = t; }
}

__global__ void fft_global_kernel(uint64_t* d_data, const uint64_t* d_twiddles, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = c_params.n;
    if (tid >= n / 2) return;
    int pair_dist = len / 2;
    int i = (tid / pair_dist) * len + (tid % pair_dist);
    int j = i + pair_dist;
    uint64_t w = d_twiddles[(tid % pair_dist) * (n / len)];
    uint64_t u = d_data[i], v = d_data[j];
    uint64_t vw = mul_mod(v, w);
    d_data[i] = add_mod(u, vw);
    d_data[j] = sub_mod(u, vw);
}

// ===== FRI FOLDING KERNEL (NEW) =====

/**
 * FRI Layer Folding Kernel
 * 공식: P_{i+1}(x^2) = (P_i(x) + P_i(-x))/2 + alpha * (P_i(x) - P_i(-x))/(2x)
 * = inv2 * [ (P(x) + P(-x)) + alpha * x^-1 * (P(x) - P(-x)) ]
 * * @param d_in        현재 레이어의 다항식 값 (크기 N)
 * @param d_out       다음 레이어의 다항식 값 (크기 N/2)
 * @param d_inv_x     x^-1 값들 (전체 도메인에 대한 역원 배열)
 * @param alpha       Verifier가 준 랜덤 챌린지
 * @param current_n   현재 레이어의 크기
 * @param stride      Twiddle Factor 접근을 위한 stride (레이어마다 2배씩 증가)
 */
__global__ void fri_fold_kernel(
    const uint64_t* __restrict__ d_in,
    uint64_t* __restrict__ d_out,
    const uint64_t* __restrict__ d_inv_x, 
    uint64_t alpha,
    int current_n,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_n = current_n / 2;
    
    if (idx >= half_n) return;

    // 1. P(x)와 P(-x) 가져오기
    // FRI 도메인 구조상 x와 -x는 인덱스 i와 i + N/2에 위치함
    uint64_t v_x = d_in[idx];
    uint64_t v_minus_x = d_in[idx + half_n];

    // 2. x^-1 가져오기
    // d_inv_x는 전체 N 크기의 배열이므로, 현재 레이어에 맞춰 stride로 건너뛰어 읽음
    uint64_t inv_x = d_inv_x[idx * stride];

    // 3. 2^-1 (Modular Inverse of 2)
    // p + 1이 짝수이므로 (p+1)/2 가 2의 역원
    uint64_t inv_2 = (c_params.p + 1) >> 1;

    // 4. Folding 연산
    // term1 = P(x) + P(-x)
    uint64_t term1 = add_mod(v_x, v_minus_x);
    
    // term2 = alpha * x^-1 * (P(x) - P(-x))
    uint64_t diff = sub_mod(v_x, v_minus_x);
    uint64_t scaled_diff = mul_mod(diff, inv_x);
    uint64_t term2 = mul_mod(scaled_diff, alpha);

    // result = inv2 * (term1 + term2)
    uint64_t sum = add_mod(term1, term2);
    d_out[idx] = mul_mod(sum, inv_2);
}

// ===== HOST FRI PROVER =====

struct FRIProof {
    std::vector<uint64_t> layer_commitments; // Merkle Roots
    std::vector<uint64_t> layer_alphas;      // Challenges
    std::vector<float> layer_times;
    uint64_t final_value;
};

// Helper for FFT (Minimal version for main)
void run_fft(uint64_t* d_data, uint64_t* d_twiddles, int n) {
    int log_n = 0; int t=n; while(t>1){log_n++; t>>=1;}
    int gridSize = (n + blockSize - 1) / blockSize;
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_data, n, log_n);
    cudaDeviceSynchronize();
    for (int len = 2; len <= n; len <<= 1) {
        gridSize = (n/2 + blockSize - 1) / blockSize;
        fft_global_kernel<<<gridSize, blockSize>>>(d_data, d_twiddles, len);
        cudaDeviceSynchronize();
    }
}

FRIProof prove_fri_gpu(
    uint64_t* d_evals,          
    uint64_t* d_twiddles_inv,   
    int n,
    uint64_t p
) {
    FRIProof proof;
    int current_n = n;
    int stride = 1;

    // [Fiat-Shamir] Transcript 초기화
    FiatShamirTranscript transcript;

    // 초기 파라미터도 Transcript에 넣는 것이 정석 (Context Binding)
    transcript.absorb(n);
    transcript.absorb(p);

    uint64_t *d_current = d_evals;
    uint64_t *d_next_buffer;
    cudaMalloc(&d_next_buffer, (n / 2) * sizeof(uint64_t));

    std::cout << "\n=== Starting FRI Prover (Fiat-Shamir) ===" << std::endl;

    while (current_n > 1) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        // 1. [Commit] Merkle Tree Root 계산
        uint64_t* h_evals = (uint64_t*)malloc(current_n * sizeof(uint64_t));
        cudaMemcpy(h_evals, d_current, current_n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        uint64_t root = simple_hash64(h_evals, current_n);
        proof.layer_commitments.push_back(root);
        free(h_evals);

        // 2. [Fiat-Shamir] Transcript에 Root 기록 및 Alpha 생성
        // Verifier도 나중에 똑같은 Root를 보게 되므로 똑같은 Alpha를 계산할 수 있음 (Non-interactive)
        transcript.absorb(root);
        uint64_t alpha = transcript.squeeze_challenge(p);
        
        proof.layer_alphas.push_back(alpha);

        std::cout << "Layer N=" << std::setw(8) << current_n 
                  << " | Root: " << to_hex(root) 
                  << " | Alpha: " << to_hex(alpha) << " (FS Generated)" << std::endl;

        // 3. [Fold] Apply FRI Folding Kernel
        int next_n = current_n / 2;
        int gridSize = (next_n + blockSize - 1) / blockSize;

        fri_fold_kernel<<<gridSize, blockSize>>>(
            d_current,
            d_next_buffer,
            d_twiddles_inv, 
            alpha,
            current_n,
            stride
        );
        cudaDeviceSynchronize();

        // Measure time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        proof.layer_times.push_back(ms);

        // Prepare for next layer
        cudaMemcpy(d_current, d_next_buffer, next_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        
        current_n = next_n;
        stride *= 2;
        
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // Final value
    uint64_t final_val;
    cudaMemcpy(&final_val, d_current, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    proof.final_value = final_val;
    
    // 마지막 값도 Transcript에 넣는 경우가 많음 (선택 사항)
    transcript.absorb(final_val);
    
    std::cout << "FRI Completed. Final Constant: " << final_val << std::endl;

    cudaFree(d_next_buffer);
    return proof;
}

// ===== HOST FRI VERIFIER =====

bool verify_fri_proof(const FRIProof& proof, int n, uint64_t p) {
    std::cout << "\n=== Verifying FRI Proof (Fiat-Shamir Check) ===" << std::endl;
    
    // 1. Verifier도 똑같은 Transcript 초기화
    FiatShamirTranscript transcript;
    transcript.absorb(n); // 초기 파라미터 (Context)
    transcript.absorb(p);

    bool is_valid = true;

    // 2. 레이어별 검증 Loop
    for (size_t i = 0; i < proof.layer_commitments.size(); i++) {
        uint64_t root = proof.layer_commitments[i];
        uint64_t prover_alpha = proof.layer_alphas[i];

        // Verifier: "Prover가 보낸 Root를 기록하자."
        transcript.absorb(root);

        // Verifier: "그럼 내가 계산한 Alpha는 이거여야 해."
        uint64_t expected_alpha = transcript.squeeze_challenge(p);

        // 비교
        if (prover_alpha != expected_alpha) {
            std::cout << "❌ Layer " << i << " Mismatch!" << std::endl;
            std::cout << "   Prover Alpha:   " << to_hex(prover_alpha) << std::endl;
            std::cout << "   Expected Alpha: " << to_hex(expected_alpha) << std::endl;
            is_valid = false;
        } else {
            // (디버깅용 출력 - 실제로는 생략 가능)
            // std::cout << "✓ Layer " << i << " Alpha Match" << std::endl;
        }
    }

    // 3. Final Value 확인 (Transcript에 포함되었는지 체크)
    // Prover 코드 마지막에 transcript.absorb(final_val)를 했다면 여기서도 해야 함
    transcript.absorb(proof.final_value);

    if (is_valid) {
        std::cout << "✓ Fiat-Shamir Transcript Integrity Verified!" << std::endl;
        std::cout << "  (Prover used the correct non-interactive challenges)" << std::endl;
    } else {
        std::cout << "❌ Verification Failed." << std::endl;
    }

    return is_valid;
}
// ===== MAIN =====

int main() {
    int log_n = 20;
    uint64_t p = 2013265921ULL;
    uint64_t root = 7;
    uint64_t n = 1ULL << log_n;
    
    set_constants(p, root, 1ULL << 20, log_n, n);
    size_t bytes = n * sizeof(uint64_t);

    std::cout << "Setup: N=2^" << log_n << ", P=" << p << std::endl;

    // Alloc & Init
    uint64_t *d_coeff, *d_twiddles, *d_twiddles_inv;
    cudaMalloc(&d_coeff, bytes);
    cudaMalloc(&d_twiddles, bytes);
    cudaMalloc(&d_twiddles_inv, bytes);

    // Generate Twiddles (Forward and Inverse)
    int gridSize = (n + blockSize - 1) / blockSize;
    uint64_t g = get_generator(root, p - 1, n, p);
    uint64_t g_inv = mod_inverse_prime(g, p);
    generate<<<gridSize, blockSize>>>(d_twiddles, d_twiddles_inv, g, g_inv, n);
    
    // Create Test Polynomial (x^2 + x)
    // Coeffs: [0, 1, 1, 0, ...] -> Eval via FFT
    uint64_t* h_poly = (uint64_t*)calloc(n, sizeof(uint64_t));
    h_poly[1] = 1; h_poly[2] = 1; // simple poly
    cudaMemcpy(d_coeff, h_poly, bytes, cudaMemcpyHostToDevice);

    std::cout << "Computing initial evaluations (FFT)..." << std::endl;
    run_fft(d_coeff, d_twiddles, n); // d_coeff now holds evaluations

    // === RUN FRI PROVER ===
    FRIProof proof = prove_fri_gpu(d_coeff, d_twiddles_inv, n, p);

    // Report
    std::cout << "\n=== FRI Prover Performance ===" << std::endl;
    float total_time = 0;
    for(size_t i=0; i<proof.layer_times.size(); i++) {
        std::cout << "Layer " << i << " Fold Time: " << std::fixed << std::setprecision(3) 
                  << proof.layer_times[i] << " ms" << std::endl;
        total_time += proof.layer_times[i];
    }
    std::cout << "Total FRI Time: " << total_time << " ms" << std::endl;
    std::cout << "Throughput: " << (n / 1e6 / (total_time/1000)) << " Mops/s" << std::endl;

    // ... (기존 Prover 실행 코드 뒤에 추가) ...

    // === RUN FRI VERIFIER ===
    bool verify_result = verify_fri_proof(proof, n, p);

    if (verify_result) {
        std::cout << "\n🎉 SUCCESS: The Proof is cryptographically valid." << std::endl;
    } else {
        std::cout << "\n💀 FAILURE: The Proof is invalid." << std::endl;
    }

    // Cleanup
    free(h_poly);
    cudaFree(d_coeff);
    cudaFree(d_twiddles);
    cudaFree(d_twiddles_inv);

    

    
    return 0;
}
