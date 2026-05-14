#pragma once
// ============================================================
//  BabyBear Montgomery Multiplication
//  p = 2013265921 = 0x78000001 = 2^31 - 2^27 + 1
//  R = 2^32  (p < 2^31 이므로 R = 2^32 으로 충분)
//
//  핵심: __int128 % p (128비트 나눗셈) 를
//        64비트 곱셈 2회 + shift + conditional subtract 로 대체
//
//  사용법:
//    uint32_t a_m = to_mont(a);      // Montgomery domain 진입
//    uint32_t c_m = mont_mul(a_m, b_m); // domain 안에서 곱셈
//    uint32_t c   = from_mont(c_m);  // domain 탈출
//
//  덧셈/뺄셈은 Montgomery와 무관 — mont_add / mont_sub 그대로 사용
// ============================================================

#include <stdint.h>
#include <cuda_runtime.h>

// ── 컴파일 타임 상수 ────────────────────────────────────────
// P        = 2013265921  (0x78000001)
// P_PRIME  = (-P^{-1}) mod 2^32 = 2013265919  (0x77FFFFFF)
//            → P * P_PRIME ≡ -1 (mod 2^32) 검증됨
// R2       = R^2 mod P = 2^64 mod P = 1172168163  (0x45DDDDE3)
//            → Montgomery domain 진입/탈출에 사용

#define MONT_P       2013265921U
#define MONT_PPRIME  2013265919U   // (-P^{-1}) mod 2^32
#define MONT_R2      1172168163U   // R^2 mod P  (R = 2^32)

// ── 핵심: Montgomery Multiplication ─────────────────────────
// REDC 알고리즘:
//   t = a * b                  (64비트)
//   m = (t mod R) * P'  mod R  (하위 32비트만)
//   u = (t + m * P) >> 32      (상위 32비트)
//   return u >= P ? u - P : u
//
// __int128 없이 uint64_t 두 번으로 완료
__device__ __host__ __forceinline__
uint32_t mont_mul(uint32_t a, uint32_t b) {
    uint64_t t = (uint64_t)a * b;
    uint32_t m = (uint32_t)(t & 0xFFFFFFFFULL) * MONT_PPRIME;
    uint64_t u = (t + (uint64_t)m * MONT_P) >> 32;
    return (u >= MONT_P) ? (uint32_t)(u - MONT_P) : (uint32_t)u;
}

// ── Montgomery domain 변환 ───────────────────────────────────
// to_mont(a)   = a * R mod P  = mont_mul(a, R²)
// from_mont(a) = a * R⁻¹ mod P = mont_mul(a, 1)
__device__ __host__ __forceinline__
uint32_t to_mont(uint32_t a) {
    return mont_mul(a, MONT_R2);
}

__device__ __host__ __forceinline__
uint32_t from_mont(uint32_t a) {
    return mont_mul(a, 1U);
}

// ── 덧셈 / 뺄셈 (Montgomery와 무관, 그대로 사용) ─────────────
__device__ __forceinline__
uint32_t mont_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return (r >= MONT_P) ? r - MONT_P : r;
}

__device__ __forceinline__
uint32_t mont_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a - b + MONT_P);
}
