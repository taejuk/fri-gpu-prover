#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <utility> // for std::pair
#include "field.h" // FieldElement 헤더

class Polynomial {
private:
    std::vector<FieldElement> poly;
    void trim();

public:
    Polynomial();
    Polynomial(const std::vector<FieldElement>& coefficients);
    Polynomial(const FieldElement& scalar);
    // 정수 (상수항) - 편의성 제공
    Polynomial(int scalar);

    static Polynomial X();
    // 단항식 생성 (coef * x^degree)
    static Polynomial monomial(int degree, const FieldElement& coefficient);
    // 일차식 생성 (x - point)
    static Polynomial gen_linear_term(const FieldElement& point);

    // --- Getter ---
    int degree() const;
    const std::vector<FieldElement>& get_coeffs() const;
    FieldElement get_nth_degree_coefficient(int n) const;

    bool operator==(const Polynomial& other) const;
    bool operator!=(const Polynomial& other) const;
    
    Polynomial operator+(const Polynomial& other) const;
    Polynomial operator-(const Polynomial& other) const;
    Polynomial operator-() const;
    
    // 다항식 곱셈
    Polynomial operator*(const Polynomial& other) const;
    
    // 스칼라 곱
    Polynomial scalar_mul(const FieldElement& scalar) const;
    Polynomial operator*(const FieldElement& scalar) const;

    // 나눗셈 (Quotient & Remainder)
    std::pair<Polynomial, Polynomial> qdiv(const Polynomial& other) const;
    Polynomial operator/(const Polynomial& other) const;
    Polynomial operator%(const Polynomial& other) const;

    Polynomial pow(long long exponent) const;
    FieldElement eval(const FieldElement& point) const;
    Polynomial compose(const Polynomial& other) const;

    // 함수 호출 연산자 오버로딩 (f(x) 문법 지원)
    FieldElement operator()(const FieldElement& point) const;
    Polynomial operator()(const Polynomial& other) const;

    // --- Lagrange Interpolation (Static Methods) ---
    // 다항식 리스트의 곱 (Divide and Conquer)
    static Polynomial poly_prod(const std::vector<Polynomial>& polys);

    // ✅ NEW: 명시적 인덱싱 버전 (OpenMP 최적화)
    static std::vector<Polynomial> calculate_lagrange_polynomials_indexed(
        const std::vector<FieldElement>& x_values);

    // 기존 버전 (유지)
    static std::vector<Polynomial> calculate_lagrange_polynomials(
        const std::vector<FieldElement>& x_values);

    static Polynomial interpolate_poly_lagrange(
        const std::vector<FieldElement>& y_values,
        const std::vector<Polynomial>& lagrange_polys);

    static Polynomial interpolate_poly(
        const std::vector<FieldElement>& x_values,
        const std::vector<FieldElement>& y_values);

    // 디버깅용 출력
    std::string toString() const;
};

inline std::ostream& operator<<(std::ostream& os, const Polynomial& p) {
    os << p.toString();
    return os;
}

// 편의성을 위한 교환법칙 지원 (FieldElement * Polynomial)
inline Polynomial operator*(const FieldElement& scalar, const Polynomial& p) {
    return p * scalar;
}

// 편의성을 위한 교환법칙 지원 (Polynomial + FieldElement)
inline Polynomial operator+(const Polynomial& p, const FieldElement& scalar) {
    return p + Polynomial(scalar);
}

inline Polynomial operator+(const FieldElement& scalar, const Polynomial& p) {
    return p + scalar;
}

#endif // POLYNOMIAL_H
