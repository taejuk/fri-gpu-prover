#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <string>



const int blockSize = 512;
const int SHARED_SIZE = 1024;
const int SHARED_STAGES = 10;


// 연산에 필요한 연산자들 정의
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

// device function
__device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t p) {
  unsigned __int128 mul = (unsigned __int128)a * b;
  return (uint64_t)(mul % p);
}

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t res = a + b;
    if (res >= p) res -= p;
    return res;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t p) {
    return (a >= b) ? (a - b) : (a - b + p);
}

// kernel function

__global__ void generate(uint64_t* d_data, uint64_t* d_data_inv, uint64_t g, uint64_t g_inv, int size, uint64_t p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        uint64_t res = 1;
        uint64_t res_inv = 1;
        uint64_t base = g;
        uint64_t base_inv = g_inv;
        int exp = idx;
    
        while(exp > 0) {
            if(exp & 1) {
                res = mul_mod(res, base, p);
                res_inv = mul_mod(res_inv, base_inv, p);
            }
            base = mul_mod(base, base, p);
            base_inv = mul_mod(base_inv, base_inv, p);
            exp = exp >> 1;
        }
        d_data[idx] = res;
        d_data_inv[idx] = res_inv;
    }
}

