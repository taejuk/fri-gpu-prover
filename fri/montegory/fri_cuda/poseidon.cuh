#pragma once
// ============================================================
//  BabyBear Poseidon12 — GPU 구현
//  p = 2013265921 = 2^31 - 2^27 + 1  (BabyBear)
//  t = 12,  alpha = 7,  RF = 8 (4+4),  RP = 22
//
//  사용 순서:
//    1. poseidon_init()              — 호스트: 상수를 constant memory에 로드
//    2. poseidon_merkle_layer<<<>>> — 커널: Merkle 레이어 1단 계산
//    3. poseidon_merkle_build()     — 헬퍼: 전체 Merkle 트리 빌드
//
//  poseidon.cuh 를 include 하는 파일 하나에만
//    #define POSEIDON_DEFINE_CONSTANTS
//  를 먼저 선언하세요 (fri_cuda.cu 맨 위).
// ============================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include "poseidon_constants.h"   // gen_poseidon_constants.py 로 생성

// ── 파라미터 ─────────────────────────────────────────────────────────────

#define POSEIDON_T            12
#define POSEIDON_RF            8   // total full rounds (4 + 4)
#define POSEIDON_RP           22
#define POSEIDON_TOTAL_ROUNDS 30   // RF + RP
#define POSEIDON_RC_SIZE     360   // TOTAL_ROUNDS * T
#define POSEIDON_MDS_SIZE    144   // T * T
#define BABYBEAR_P  2013265921ULL

// ── Constant memory 선언 ─────────────────────────────────────────────────
// .cu 파일 하나에서만 POSEIDON_DEFINE_CONSTANTS 를 define해서 실체화.
// 나머지 translation unit 은 extern 선언만 참조.

#ifdef POSEIDON_DEFINE_CONSTANTS
__constant__ uint64_t d_poseidon_rc [POSEIDON_RC_SIZE];
__constant__ uint64_t d_poseidon_mds[POSEIDON_MDS_SIZE];
#else
extern __constant__ uint64_t d_poseidon_rc [POSEIDON_RC_SIZE];
extern __constant__ uint64_t d_poseidon_mds[POSEIDON_MDS_SIZE];
#endif

// ── 호스트 초기화 ─────────────────────────────────────────────────────────

inline cudaError_t poseidon_init() {
    cudaError_t e;
    e = cudaMemcpyToSymbol(d_poseidon_rc,
                           POSEIDON_RC,
                           sizeof(uint64_t) * POSEIDON_RC_SIZE);
    if (e != cudaSuccess) return e;
    return cudaMemcpyToSymbol(d_poseidon_mds,
                              POSEIDON_MDS,
                              sizeof(uint64_t) * POSEIDON_MDS_SIZE);
}

// ── 디바이스 필드 연산 (BabyBear 특화, c_params 불필요) ──────────────────

__device__ __forceinline__ uint64_t bb_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r >= BABYBEAR_P) r -= BABYBEAR_P;
    return r;
}

__device__ __forceinline__ uint64_t bb_mul(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a * b) % BABYBEAR_P);
}

// x^7 = x * x^2 * x^4  — 곱셈 3회
__device__ __forceinline__ uint64_t bb_pow7(uint64_t x) {
    uint64_t x2 = bb_mul(x, x);
    uint64_t x4 = bb_mul(x2, x2);
    return bb_mul(bb_mul(x, x2), x4);
}

// ── MDS 행렬-벡터 곱 ─────────────────────────────────────────────────────
// state' = MDS × state  (mod p)
// d_poseidon_mds 는 row-major [T × T]

__device__ __forceinline__ void mds_multiply(uint64_t state[POSEIDON_T]) {
    uint64_t tmp[POSEIDON_T];

    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        uint64_t acc = 0;
        #pragma unroll
        for (int j = 0; j < POSEIDON_T; j++) {
            acc = bb_add(acc,
                         bb_mul(d_poseidon_mds[i * POSEIDON_T + j], state[j]));
        }
        tmp[i] = acc;
    }
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) state[i] = tmp[i];
}

// ── Poseidon12 퍼뮤테이션 ─────────────────────────────────────────────────
// 구조: [RF/2 full] → [RP partial] → [RF/2 full]
// 각 라운드: AddRoundConstants → SubWords → MixLayer

__device__ void poseidon_permutation(uint64_t state[POSEIDON_T]) {
    int rc = 0;  // round constant 인덱스

    // ── 앞쪽 full rounds (4회) ──────────────────────────────────────────
    #pragma unroll
    for (int r = 0; r < POSEIDON_RF / 2; r++) {
        // AddRoundConstants
        #pragma unroll
        for (int i = 0; i < POSEIDON_T; i++)
            state[i] = bb_add(state[i], d_poseidon_rc[rc++]);
        // SubWords — 전체 T개
        #pragma unroll
        for (int i = 0; i < POSEIDON_T; i++)
            state[i] = bb_pow7(state[i]);
        // MixLayer
        mds_multiply(state);
    }

    // ── Partial rounds (22회) ────────────────────────────────────────────
    for (int r = 0; r < POSEIDON_RP; r++) {
        // AddRoundConstants
        #pragma unroll
        for (int i = 0; i < POSEIDON_T; i++)
            state[i] = bb_add(state[i], d_poseidon_rc[rc++]);
        // SubWords — state[0] 만
        state[0] = bb_pow7(state[0]);
        // MixLayer
        mds_multiply(state);
    }

    // ── 뒤쪽 full rounds (4회) ──────────────────────────────────────────
    #pragma unroll
    for (int r = 0; r < POSEIDON_RF / 2; r++) {
        #pragma unroll
        for (int i = 0; i < POSEIDON_T; i++)
            state[i] = bb_add(state[i], d_poseidon_rc[rc++]);
        #pragma unroll
        for (int i = 0; i < POSEIDON_T; i++)
            state[i] = bb_pow7(state[i]);
        mds_multiply(state);
    }
    // rc == POSEIDON_RC_SIZE (360) 이어야 정상
}

// ── 2-to-1 압축 함수 ─────────────────────────────────────────────────────
// sponge 방식: state = [left, right, 0, ..., 0] → permute → state[0] 반환
// state[2..11] = 0 이 capacity 역할을 해 도메인 분리

__device__ __forceinline__ uint64_t poseidon_compress(uint64_t left, uint64_t right) {
    uint64_t state[POSEIDON_T] = {};  // 전부 0으로 초기화
    state[0] = left;
    state[1] = right;
    poseidon_permutation(state);
    return state[0];
}

// ── Merkle 레이어 커널 ────────────────────────────────────────────────────
// d_in  [2 * n_pairs] : 현재 레이어 노드 (짝수 인덱스=왼쪽, 홀수=오른쪽)
// d_out [n_pairs]     : 부모 노드
// 스레드 1개 = 노드 쌍 1개 처리

__global__ void poseidon_merkle_layer(
    const uint64_t* __restrict__ d_in,
    uint64_t*       __restrict__ d_out,
    int n_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;
    d_out[idx] = poseidon_compress(d_in[2 * idx], d_in[2 * idx + 1]);
}

// ── 전체 Merkle 트리 빌드 헬퍼 ───────────────────────────────────────────
// n 은 2의 거듭제곱이어야 함.
// 핑퐁 버퍼 방식 — in-place 는 race condition 발생하므로 금지.
//
// 반환: 루트 값 (*h_root 에 저장, 호스트 포인터)

inline cudaError_t poseidon_merkle_build(
    const uint64_t* d_leaves,  // 디바이스 입력 (n개)
    uint64_t*       d_buf0,    // 디바이스 버퍼 (n개)  ← 호출자 할당
    uint64_t*       d_buf1,    // 디바이스 버퍼 (n개)  ← 호출자 할당
    uint64_t*       h_root,    // 호스트 출력 (1개)
    int             n,
    int             block_size = 256)
{
    cudaError_t e;

    // 잎 → buf0 복사
    e = cudaMemcpy(d_buf0, d_leaves, n * sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) return e;

    uint64_t* src = d_buf0;
    uint64_t* dst = d_buf1;
    int cur = n;

    while (cur > 1) {
        int pairs = cur / 2;
        int grid  = (pairs + block_size - 1) / block_size;
        poseidon_merkle_layer<<<grid, block_size>>>(src, dst, pairs);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        cudaDeviceSynchronize();

        // 핑퐁
        uint64_t* tmp = src; src = dst; dst = tmp;
        cur = pairs;
    }
    // 루트는 src[0]
    return cudaMemcpy(h_root, src, sizeof(uint64_t), cudaMemcpyDeviceToHost);
}
