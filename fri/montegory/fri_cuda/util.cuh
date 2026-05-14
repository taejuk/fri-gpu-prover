#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <string>
#include "mont.cuh"

const int blockSize = 512;
const int SHARED_SIZE = 1024;
const int SHARED_STAGES = 10;

// ── 호스트 연산 (변경 없음, __int128 그대로) ─────────────────
// 호스트 코드는 mod_inverse_prime, get_generator 등 일회성 계산이라
// 성능 무관 — 변경 불필요

uint64_t multiply_uint64(uint64_t a, uint64_t b, uint64_t p) {
    unsigned __int128 mul = (unsigned __int128)a * b;
    return (uint64_t)(mul % p);
}

uint64_t mod_inverse_prime(uint64_t a, uint64_t p) {
    uint64_t result = 1, base = a, exp = p - 2;
    while (exp > 0) {
        if (exp & 1) result = ((__uint128_t)result * base) % p;
        base = ((__uint128_t)base * base) % p;
        exp >>= 1;
    }
    return result;
}

uint64_t get_generator(uint64_t g, uint64_t n, uint64_t m, uint64_t p) {
    uint64_t pw = n / m, res = 1, base = g;
    while (pw > 0) {
        if (pw & 1) res = multiply_uint64(res, base, p);
        base = multiply_uint64(base, base, p);
        pw >>= 1;
    }
    return res;
}

// ── 디바이스 필드 연산 — Montgomery 버전 ─────────────────────
// 기존: __int128 % p  →  새: mont_mul (64비트 연산만)
// 타입을 uint32_t 로 통일 (BabyBear 원소는 31비트)
// ntt/intt 커널은 uint32_t 배열을 사용하도록 변경

__device__ __forceinline__
uint32_t mul_mod(uint32_t a, uint32_t b) {
    return mont_mul(a, b);   // 호출부 시그니처 유지 (p 인자 제거)
}

__device__ __forceinline__
uint32_t add_mod(uint32_t a, uint32_t b) {
    return mont_add(a, b);
}

__device__ __forceinline__
uint32_t sub_mod(uint32_t a, uint32_t b) {
    return mont_sub(a, b);
}

// ── generate 커널: twiddle factor를 Montgomery 형태로 저장 ─────
// d_data[i]     = to_mont(g^i mod p)
// d_data_inv[i] = to_mont(g_inv^i mod p)
// 커널 내부는 Montgomery domain에서 곱셈하므로
// mont_mul(a_m, b_m) = (a*b*R^{-1} mod p) * R = (a*b) mod p 의 mont 형태
// → 입력이 mont 형태면 출력도 mont 형태

__global__ void generate(uint32_t* d_data, uint32_t* d_data_inv,
                          uint32_t g_m, uint32_t g_inv_m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Montgomery domain 안에서 g_m^idx 계산 (binary exponentiation)
    uint32_t res     = to_mont(1U);   // 1의 Montgomery 형태 = R mod p
    uint32_t res_inv = to_mont(1U);
    uint32_t base     = g_m;
    uint32_t base_inv = g_inv_m;
    int exp = idx;

    while (exp > 0) {
        if (exp & 1) {
            res     = mont_mul(res,     base);
            res_inv = mont_mul(res_inv, base_inv);
        }
        base     = mont_mul(base,     base);
        base_inv = mont_mul(base_inv, base_inv);
        exp >>= 1;
    }
    d_data[idx]     = res;
    d_data_inv[idx] = res_inv;
}

// ── uint32_t → uint64_t 확장 커널 ────────────────────────────
// NTT 결과(uint32_t)를 FRI Poseidon(uint64_t) 입력으로 변환
__global__ void expand_u32_to_u64(const uint32_t* __restrict__ src,
                                   uint64_t* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = (uint64_t)src[idx];
}

// ── twiddle만 생성 (inverse 불필요할 때) ─────────────────────
__global__ void generateWithoutInv(uint32_t* d_data, uint32_t g_m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    uint32_t res  = to_mont(1U);
    uint32_t base = g_m;
    int exp = idx;

    while (exp > 0) {
        if (exp & 1) res = mont_mul(res, base);
        base = mont_mul(base, base);
        exp >>= 1;
    }
    d_data[idx] = res;
}
