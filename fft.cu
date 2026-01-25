#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
// 1. 에러 체크 매크로 (필수: CUDA는 에러가 나도 조용히 넘어가기 때문)
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}


struct KernelParams {
    uint64_t p;
    uint64_t root;
    uint64_t root_pw;
    int log_n;
};

__constant__ KernelParams device_params;
// mod p에 대한 곱셈이다. 이 때 p = 2^64 - 2^31 + 1 이므로 
__device__ uint64_t  multiply_uint64(uint64_t a, uint64_t b) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)mul;
    uint64_t hi = (uint64_t)(mul >> 64);
    uint64_t hi_shifted = hi << 32;
    uint64_t res = lo - hi + hi_shifted;

    
    const uint64_t EPSILON = 0xFFFFFFFF;
    if (lo < hi) res -= EPSILON;
    if (res >= device_params.p) res -= device_params.p;

    return res;	
}

uint64_t multiply_uint64(uint64_t a, uint64_t b, uint64_t p) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)mul;
    uint64_t hi = (uint64_t)(mul >> 64);
    uint64_t hi_shifted = hi << 32;
    uint64_t res = lo - hi + hi_shifted;


    const uint64_t EPSILON = 0xFFFFFFFF;
    if (lo < hi) res -= EPSILON;
    if (res >= p) res -= p;

    return res;
}



void set_constants(uint64_t p, uint64_t root, uint64_t root_pw, uint64_t log_n) {
    KernelParams host_params = {p, root, root_pw, log_n};
    cudaMemcpyToSymbol(device_params, &host_params, sizeof(KernelParams));
}

template <typename T>
__device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}


__device__ int reverse(int idx) {
    int res = 0;
    for (int i = 0; i < device_params.log_n; i++) {
        if (idx & (1 << i))
            res |= 1 << (device_params.log_n - 1 - i);
    }
    return res;

}

// 
__global__ void reallocate(uint64_t* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx를 받아서 계수 swap하기
    int newidx = reverse(idx);
    if(idx < newidx) swap(d_data[idx], d_data[newidx]);
}
// 이 함수는 계수의 반만큼만 실행하면 된다. 왜냐하면 하나의 for문에 데이터 두개를 바꾸기 때문
// 이렇게 하면 의존성을 크게 신경 안쓰고 할 수 있다.
// 내가 교체하는 index인지 아닌지만 판단하면 된다.
__global__ void calculate(uint64_t* d_data, int n, uint64_t wlen, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx % len;
    if(j < len/2) {
    	uint64_t w = 1;
	for(int i = 0; i < j ; i++) w = multiply_uint64(w, wlen);
	uint64_t u = d_data[idx]; uint64_t v = multiply_uint64(d_data[idx + len/2], w);
	d_data[idx] = u + v < device_params.p ? u + v : u + v - device_params.p;
	d_data[idx + len/2] = u - v < device_params.p ? u - v : u - v + device_params.p;
    }
}


void print_binary_custom(uint64_t n) {
    std::cout << "Binary: ";
    bool leading_zeros = false; // 앞쪽 0 생략 여부

    // 63번째 비트부터 0번째 비트까지 검사
    for (int i = 19; i >= 0; i--) {
        // i번째 비트가 1인지 확인
        uint64_t bit = (n >> i) & 1;

        // (옵션) 앞쪽 불필요한 0 생략 로직
        if (leading_zeros && bit == 0 && i != 0) continue;
        leading_zeros = false;

        std::cout << bit;

        // (옵션) 가독성을 위해 4자리마다 공백 추가
        if (i % 4 == 0 && i != 0) std::cout << " ";
    }
    std::cout << std::endl;
}



int main() {
    int log_n = 20;
    // 2^64 - 2^31 + 1
    uint64_t p = 0xFFFFFFFF00000001;
    uint64_t root = 1753635133440165772ULL;
    uint64_t root_pw = 1ULL << 32;;
    set_constants(p, root, root_pw, log_n); 
    int N = 1 << log_n; // 2^20 (약 100만 개 데이터)
    size_t bytes = N * sizeof(uint64_t); // 64비트 데이터 크기 계산

    std::cout << "Data Size: " << N << " elements (" << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    // === 3. Host(CPU) 메모리 할당 및 초기화 ===
    // [최적화 Tip]: 그냥 new 대신 cudaMallocHost를 쓰면 'Pinned Memory'가 되어 전송이 2배 빨라짐
    uint64_t* h_data;
    CHECK_CUDA(cudaMallocHost((void**)&h_data, bytes)); 

    // 데이터 채우기 (0, 1, 2, ...)
    for (int i = 0; i < N; i++) {
        h_data[i] = 0;
    }
    h_data[0] = 1;
    h_data[1] = 2;

    // === 4. Device(GPU) 메모리 할당 ===
    uint64_t* d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));

    // === 5. 데이터 전송 (Host -> Device) ===
    std::cout << "🚀 Transferring data to GPU..." << std::endl;
    
    // cudaMemcpy(목적지, 출발지, 크기, 방향)
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // === 6. 커널 실행 (잘 갔는지 테스트) ===
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    std::cout << gridSize << " " << blockSize << std::endl;
    reallocate<<<gridSize, blockSize>>>(d_data, N);
    
    for(int len = 2; len <= N; len <<=1) {
	uint64_t wlen = root;
    	for(int i = len; i < root_pw; i <<= 1) wlen = multiply_uint64(wlen, wlen, p);
	//calculate(uint64_t* d_data, int n, uint64_t wlen, int len)
	calculate<<<gridSize, blockSize>>>(d_data, N, wlen, len);
    }

    // len을 기준으로 계속 반복한다.
    CHECK_CUDA(cudaGetLastError()); // 커널 실행 에러 체크
    CHECK_CUDA(cudaDeviceSynchronize()); // GPU 작업 끝날 때까지 대기

    // === 7. 결과 확인 (Device -> Host) ===
    // 확인을 위해 다시 CPU로 가져옴
    std::cout << "📥 Bringing data back to CPU..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    //for (int i = 0; i < 10; i++) { // 앞부분 10개만 출력
    //    printf("Index %d: Expected %llu, Got %llu\n", i, (uint64_t)i + 1, h_data[i]);
    //    if (h_data[i] != i + 1) correct = false;
    //}
    for (int i = 0; i < 10; i++) {
    	//print_binary_custom(h_data[i]);
	std::cout << h_data[i] << std::endl;
    }
    //if (correct) std::cout << "✅ Test PASSED!" << std::endl;
    //else std::cout << "❌ Test FAILED!" << std::endl;

    // === 8. 메모리 해제 ===
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFreeHost(h_data)); // cudaMallocHost로 할당했으므로 이걸로 해제

    return 0;
}
