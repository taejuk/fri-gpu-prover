#ifndef NTT_H
#define NTT_H

#include "field.h"
#include "polynomial.h"
#include <vector>
#include <algorithm>
#include <cmath>

class NTT {
private:
    static const int128 MODULUS = 2013265921ULL;
    static const int128 GENERATOR = 7;
    
    // Bit-Reversal Permutation
    // n을 이진 표현으로 뒤집은 값을 반환
    static int bitReverse(int n, int bits) {
        int result = 0;
        for (int i = 0; i < bits; i++) {
            result = (result << 1) | (n & 1);
            n >>= 1;
        }
        return result;
    }

    // Primitive n-th root of unity 계산
    // omega = generator^((p-1)/n)
    static FieldElement getPrimitiveRoot(int n) {
        // (MODULUS - 1) = 2^27 * 15 이므로
        // 2^20 크기의 NTT를 하려면: exponent = (MODULUS - 1) / 2^20
        int128 exponent = (MODULUS - 1) / n;
        return FieldElement(GENERATOR).pow(exponent);
    }

    // In-place Cooley-Tukey NTT (Radix-2)
    static void nttRecursive(std::vector<FieldElement>& a, bool invert) {
        int n = a.size();
        if (n == 1) return;

        // Bit-reversal permutation
        int bits = 0;
        int temp = n - 1;
        while (temp > 0) {
            bits++;
            temp >>= 1;
        }

        for (int i = 0; i < n; i++) {
            int j = bitReverse(i, bits);
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }

        // Butterfly operations (iterative)
        for (int len = 2; len <= n; len <<= 1) {
            FieldElement omega = getPrimitiveRoot(len);
            
            if (invert) {
                // Inverse NTT: omega^(-1)
                omega = omega.inverse();
            }

            for (int i = 0; i < n; i += len) {
                FieldElement w(1);
                for (int j = 0; j < len / 2; j++) {
                    FieldElement u = a[i + j];
                    FieldElement v = a[i + j + len / 2] * w;
                    
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    
                    w = w * omega;
                }
            }
        }

        // Normalization for inverse transform
        if (invert) {
            FieldElement n_inv = FieldElement(n).inverse();
            for (auto& x : a) {
                x = x * n_inv;
            }
        }
    }

public:
    // Forward NTT: coefficients -> values at roots of unity
    static void forward(std::vector<FieldElement>& a) {
        nttRecursive(a, false);
    }

    // Inverse NTT: values at roots of unity -> coefficients
    static void inverse(std::vector<FieldElement>& a) {
        nttRecursive(a, true);
    }

    // Polynomial evaluation using NTT
    // poly: Polynomial to evaluate
    // domain: evaluation points (should be n-th roots of unity, n = power of 2)
    // Returns: evaluations of poly at domain points
    static std::vector<FieldElement> evalDomain(
        const Polynomial& poly,
        const std::vector<FieldElement>& domain) {
        
        int n = domain.size();
        
        // Check if n is power of 2
        if ((n & (n - 1)) != 0) {
            throw std::invalid_argument("Domain size must be a power of 2");
        }

        // Get polynomial coefficients and pad to domain size
        std::vector<FieldElement> coeffs = poly.get_coeffs();
        coeffs.resize(n, FieldElement(0));

        // Forward NTT
        forward(coeffs);
        
        return coeffs;
    }

    // Polynomial interpolation using inverse NTT
    // evaluations: values of polynomial at n-th roots of unity
    // Returns: Polynomial with coefficients recovered from evaluations
    static Polynomial interpolatePoly(
        const std::vector<FieldElement>& evaluations) {
        
        int n = evaluations.size();
        
        // Check if n is power of 2
        if ((n & (n - 1)) != 0) {
            throw std::invalid_argument("Evaluation size must be a power of 2");
        }

        std::vector<FieldElement> coeffs = evaluations;
        
        // Inverse NTT
        inverse(coeffs);
        
        return Polynomial(coeffs);
    }

    // Test primitive root of unity properties
    static bool testPrimitiveRoot(int n) {
        FieldElement omega = getPrimitiveRoot(n);
        
        // Test: omega^n = 1
        FieldElement omega_n = omega.pow(n);
        if (omega_n != FieldElement(1)) {
            return false;
        }
        
        // Test: omega^k != 1 for 0 < k < n
        for (int k = 1; k < n; k++) {
            if (omega.pow(k) == FieldElement(1)) {
                return false;
            }
        }
        
        return true;
    }
};

#endif // NTT_H