#ifndef FIELD_H
#define FIELD_H
#include <string>
#include <algorithm>
using int128 = unsigned __int128;

class FieldElement {
public:
    

    static const int128 k_modulus = 2013265921ULL;
    static const int128 generator_val = 7;

    int128 val;

    // 생성자
    FieldElement(int128 v = 0) {
        val = (v % k_modulus + k_modulus) % k_modulus;
    }

    static FieldElement zero() { return FieldElement(0); }
    static FieldElement one() { return FieldElement(1); }
    static FieldElement generator() { return FieldElement(generator_val); }

    // 연산자 오버로딩
    bool operator==(const FieldElement& other) const { return val == other.val; }
    bool operator!=(const FieldElement& other) const { return val != other.val; }

    FieldElement operator-() const {
        return FieldElement(k_modulus - val);
    }

    FieldElement operator+(const FieldElement& other) const {
        return FieldElement((val + other.val) % k_modulus);
    }

    FieldElement operator-(const FieldElement& other) const {
        return FieldElement((val - other.val + k_modulus) % k_modulus);
    }

    FieldElement operator*(const FieldElement& other) const {
         return FieldElement(((__uint128_t)val * (__uint128_t)other.val) % k_modulus);
    }

    FieldElement operator/(const FieldElement& other) const {
        return (*this) * other.inverse();
    }

    FieldElement pow(unsigned long long n) const {
        FieldElement res(1);
        FieldElement cur_pow = *this;
        while (n > 0) {
            if (n % 2 != 0) res = res * cur_pow;
            cur_pow = cur_pow * cur_pow;
            n /= 2;
        }
        return res;
    }

    FieldElement inverse() const {
        long long t = 0, new_t = 1;  // ← signed 타입 사용
        long long r = (long long)k_modulus, new_r = (long long)val;
        
        while (new_r != 0) {
            long long quotient = r / new_r;
            long long temp_t = t;
            t = new_t;
            new_t = temp_t - quotient * new_t;
            
            long long temp_r = r;
            r = new_r;
            new_r = temp_r - quotient * new_r;
        }
        
        if (r > 1) throw std::runtime_error("Value is not invertible");
        if (t < 0) t += (long long)k_modulus;  // ← 이제 정상 작동
        
        return FieldElement((int128)t);
    }


    std::string to_string() const {

        return int128ToString(val);
    }


private:
    static std::string int128ToString(int128 n) {
        if (n == 0) return "0";
        std::string s = "";
        while (n > 0) {
            s += (char)('0' + (n % 10));
            n /= 10;
        }
        std::reverse(s.begin(), s.end());
        return s;
    }
};

inline std::ostream& operator<<(std::ostream& os, const FieldElement& fe) {
    os << fe.to_string();
    return os;
}

#endif