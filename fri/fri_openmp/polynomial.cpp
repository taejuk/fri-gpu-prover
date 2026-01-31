#include "polynomial.h"

// --- Private Helper ---
void Polynomial::trim() {
    while (!poly.empty() && poly.back() == FieldElement::zero()) {
        poly.pop_back();
    }
}

// --- Constructors ---
Polynomial::Polynomial() {}

Polynomial::Polynomial(const std::vector<FieldElement>& coefficients) : poly(coefficients) {
    trim();
}

Polynomial::Polynomial(const FieldElement& scalar) {
    poly.push_back(scalar);
    trim();
}

Polynomial::Polynomial(int scalar) {
    poly.push_back(FieldElement(scalar));
    trim();
}

// --- Static Factory Methods ---
Polynomial Polynomial::X() {
    std::vector<FieldElement> coeffs(2, FieldElement::zero());
    coeffs[1] = FieldElement::one();
    return Polynomial(coeffs);
}

Polynomial Polynomial::monomial(int degree, const FieldElement& coefficient) {
    if (coefficient == FieldElement::zero()) return Polynomial();
    std::vector<FieldElement> coeffs(degree + 1, FieldElement::zero());
    coeffs[degree] = coefficient;
    return Polynomial(coeffs);
}

Polynomial Polynomial::gen_linear_term(const FieldElement& point) {
    std::vector<FieldElement> coeffs(2, FieldElement::zero());
    coeffs[0] = FieldElement::zero() - point;
    coeffs[1] = FieldElement::one();
    return Polynomial(coeffs);
}

// --- Getters ---
int Polynomial::degree() const {
    return static_cast<int>(poly.size()) - 1;
}

const std::vector<FieldElement>& Polynomial::get_coeffs() const {
    return poly;
}

FieldElement Polynomial::get_nth_degree_coefficient(int n) const {
    if (n < 0 || n >= static_cast<int>(poly.size())) return FieldElement::zero();
    return poly[n];
}

// --- Operators ---
bool Polynomial::operator==(const Polynomial& other) const {
    return poly == other.poly;
}

bool Polynomial::operator!=(const Polynomial& other) const {
    return poly != other.poly;
}

Polynomial Polynomial::operator+(const Polynomial& other) const {
    std::vector<FieldElement> res(std::max(poly.size(), other.poly.size()));
    for (size_t i = 0; i < res.size(); ++i) {
        FieldElement a = (i < poly.size()) ? poly[i] : FieldElement::zero();
        FieldElement b = (i < other.poly.size()) ? other.poly[i] : FieldElement::zero();
        res[i] = a + b;
    }
    return Polynomial(res);
}

Polynomial Polynomial::operator-(const Polynomial& other) const {
    std::vector<FieldElement> res(std::max(poly.size(), other.poly.size()));
    for (size_t i = 0; i < res.size(); ++i) {
        FieldElement a = (i < poly.size()) ? poly[i] : FieldElement::zero();
        FieldElement b = (i < other.poly.size()) ? other.poly[i] : FieldElement::zero();
        res[i] = a - b;
    }
    return Polynomial(res);
}

Polynomial Polynomial::operator-() const {
    std::vector<FieldElement> res;
    res.reserve(poly.size());
    for (const auto& c : poly) res.push_back(-c);
    return Polynomial(res);
}

Polynomial Polynomial::operator*(const Polynomial& other) const {
    if (poly.empty() || other.poly.empty()) return Polynomial();
    
    std::vector<FieldElement> res(poly.size() + other.poly.size() - 1, FieldElement::zero());
    for (size_t i = 0; i < poly.size(); ++i) {
        for (size_t j = 0; j < other.poly.size(); ++j) {
            res[i + j] = res[i + j] + (poly[i] * other.poly[j]);
        }
    }
    return Polynomial(res);
}

Polynomial Polynomial::scalar_mul(const FieldElement& scalar) const {
    if (scalar == FieldElement::zero()) return Polynomial();
    std::vector<FieldElement> res;
    res.reserve(poly.size());
    for (const auto& c : poly) res.push_back(c * scalar);
    return Polynomial(res);
}

Polynomial Polynomial::operator*(const FieldElement& scalar) const {
    return scalar_mul(scalar);
}

// --- Division ---
std::pair<Polynomial, Polynomial> Polynomial::qdiv(const Polynomial& other) const {
    if (other.poly.empty()) throw std::runtime_error("Division by zero polynomial");
    if (poly.empty()) return std::make_pair(Polynomial(), Polynomial());

    std::vector<FieldElement> rem = poly;
    const std::vector<FieldElement>& divisor = other.poly;
    
    int deg_rem = static_cast<int>(rem.size()) - 1;
    int deg_div = static_cast<int>(divisor.size()) - 1;
    int deg_dif = deg_rem - deg_div;

    if (deg_dif < 0) {
        return std::make_pair(Polynomial(), *this);
    }

    std::vector<FieldElement> quotient(deg_dif + 1, FieldElement::zero());
    FieldElement div_lead_inv = divisor.back().inverse();

    for (int i = deg_dif; i >= 0; --i) {
        if (static_cast<int>(rem.size()) - 1 < i + deg_div) continue;

        // 수정: rem의 인덱스는 i + deg_div가 맞습니다.
        FieldElement q = rem.back() * div_lead_inv; 
        quotient[i] = q;

        for (size_t j = 0; j < divisor.size(); ++j) {
            rem[i + j] = rem[i + j] - (q * divisor[j]);
        }

        while (!rem.empty() && rem.back() == FieldElement::zero()) {
            rem.pop_back();
        }
    }

    return std::make_pair(Polynomial(quotient), Polynomial(rem));
}

Polynomial Polynomial::operator/(const Polynomial& other) const {
    auto result = qdiv(other);
    // 나눗셈 후 나머지가 0인지 확인하는 로직 (선택적)
    // if (result.second != Polynomial()) throw std::runtime_error("Polynomials are not divisible");
    return result.first;
}

Polynomial Polynomial::operator%(const Polynomial& other) const {
    return qdiv(other).second;
}

// --- Math & Eval ---
Polynomial Polynomial::pow(long long exponent) const {
    if (exponent < 0) throw std::invalid_argument("Exponent must be non-negative");
    Polynomial res(1);
    Polynomial base = *this;
    while (exponent > 0) {
        if (exponent % 2 == 1) res = res * base;
        base = base * base;
        exponent /= 2;
    }
    return res;
}

FieldElement Polynomial::eval(const FieldElement& point) const {
    FieldElement val = FieldElement::zero();
    for (auto it = poly.rbegin(); it != poly.rend(); ++it) {
        val = val * point + *it;
    }
    return val;
}

Polynomial Polynomial::compose(const Polynomial& other) const {
    Polynomial res;
    for (auto it = poly.rbegin(); it != poly.rend(); ++it) {
        res = (res * other) + Polynomial(*it);
    }
    return res;
}

FieldElement Polynomial::operator()(const FieldElement& point) const {
    return eval(point);
}

Polynomial Polynomial::operator()(const Polynomial& other) const {
    return compose(other);
}

// --- Interpolation (Static) ---
Polynomial Polynomial::poly_prod(const std::vector<Polynomial>& polys) {
    if (polys.empty()) return Polynomial(FieldElement::one());
    if (polys.size() == 1) return polys[0];

    size_t mid = polys.size() / 2;
    std::vector<Polynomial> left(polys.begin(), polys.begin() + mid);
    std::vector<Polynomial> right(polys.begin() + mid, polys.end());

    return poly_prod(left) * poly_prod(right);
}

std::vector<Polynomial> Polynomial::calculate_lagrange_polynomials(const std::vector<FieldElement>& x_values) {
    std::vector<Polynomial> lagrange_polys;
    lagrange_polys.reserve(x_values.size());

    std::vector<Polynomial> monomials;
    monomials.reserve(x_values.size());
    for (const auto& x : x_values) {
        monomials.push_back(Polynomial::X() - Polynomial(x));
    }
    Polynomial numerator = poly_prod(monomials);

    for (size_t j = 0; j < x_values.size(); ++j) {
        FieldElement denominator = FieldElement::one();
        for (size_t i = 0; i < x_values.size(); ++i) {
            if (i != j) {
                denominator = denominator * (x_values[j] - x_values[i]);
            }
        }
        Polynomial divisor = monomials[j] * denominator;
        lagrange_polys.push_back(numerator.qdiv(divisor).first);
    }
    return lagrange_polys;
}

Polynomial Polynomial::interpolate_poly_lagrange(const std::vector<FieldElement>& y_values,
                                            const std::vector<Polynomial>& lagrange_polys) {
    if (y_values.size() != lagrange_polys.size()) throw std::invalid_argument("Size mismatch");
    
    Polynomial res;
    for (size_t i = 0; i < y_values.size(); ++i) {
        res = res + (lagrange_polys[i] * y_values[i]);
    }
    return res;
}

Polynomial Polynomial::interpolate_poly(const std::vector<FieldElement>& x_values,
                                   const std::vector<FieldElement>& y_values) {
    if (x_values.size() != y_values.size()) throw std::invalid_argument("Size mismatch");
    auto lagrange_polys = calculate_lagrange_polynomials(x_values);
    return interpolate_poly_lagrange(y_values, lagrange_polys);
}

std::string Polynomial::toString() const {
    if (poly.empty()) return "0";
    std::string s = "";
    for (int i = degree(); i >= 0; --i) {
        if (poly[i] == FieldElement::zero()) continue;
        if (!s.empty()) s += " + ";
        s += poly[i].to_string();
        if (i > 0) s += "x^" + std::to_string(i);
    }
    return s.empty() ? "0" : s;
}