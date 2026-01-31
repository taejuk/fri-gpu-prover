#include <iostream>
#include <vector>
#include "polynomial.h"
#include "merkle.h"
#include "ntt.h"
#include <chrono>
#include <string>
#include <iomanip>

using namespace std;
// generator 생성용
FieldElement get_primitive_nth_root(int n) {
    int128 exponent = (FieldElement::k_modulus - (int128)1) / (int128)n;
    return FieldElement(7).pow(exponent);
}

// Naive evaluation (O(n^2)) - 비교용
vector<FieldElement> eval_domain_naive(const Polynomial& poly, const vector<FieldElement>& domain) {
    vector<FieldElement> res;
    for(size_t i = 0; i < domain.size(); i++) {
        res.push_back(poly(domain[i]));
    }
    return res;
}

// NTT 기반 평가 (O(n log n))
vector<FieldElement> eval_domain_ntt(const Polynomial& poly, const vector<FieldElement>& domain) {
    return NTT::evalDomain(poly, domain);
}

// FRI commitment 구조 (간단한 버전)
struct FRICommitment {
    vector<string> layer_hashes;  // 각 레이어의 Merkle root
    vector<int> layer_sizes;      // 각 레이어의 크기
};

// FRI commitment 생성
FRICommitment fri_commitment(
    const Polynomial& poly,
    const vector<FieldElement>& domain,
    int num_layers) {
    
    FRICommitment result;
    vector<FieldElement> current_evaluations = eval_domain_ntt(poly, domain);
    
    cout << "FRI Layers:" << endl;
    
    for(int layer = 0; layer < num_layers; layer++) {
        cout << "  Layer " << layer << ": " << current_evaluations.size() << " points";
        
        // Merkle tree 구축
        MerkleTree tree(current_evaluations);
        string root = tree.get_root();
        
        result.layer_hashes.push_back(root);
        result.layer_sizes.push_back(current_evaluations.size());
        
        cout << " -> root: " << root.substr(0, 16) << "..." << endl;
        
        if(current_evaluations.size() <= 1) break;
        
        // 다음 라운드: 도메인을 제곱 (fold)
        // FRI folding: f(x) + f(-x) / 2를 다음 라운드의 polynomial로 사용
        // 간단화: 현재는 크기만 절반으로 줄임
        vector<FieldElement> next_evals(current_evaluations.size() / 2);
        for(size_t i = 0; i < next_evals.size(); i++) {
            // Simple folding: 인접한 두 값의 평균 (실제 FRI는 더 복잡함)
            next_evals[i] = (current_evaluations[2*i] + current_evaluations[2*i+1]) * 
                           FieldElement(2).inverse();  // /2 in field
        }
        current_evaluations = next_evals;
    }
    
    return result;
}

int main() {
    cout << "=== FRI Commitment with NTT ===" << endl << endl;
    
    auto total_start = chrono::high_resolution_clock::now();
    
    // 파라미터 설정
    int log_size = 20;  // 2^20 = 1,048,576
    int poly_size = (1 << log_size);
    
    cout << "Polynomial size: 2^" << log_size << " = " << poly_size << " coefficients" << endl;
    
    // 계수 및 도메인 생성
    vector<FieldElement> coeffs(poly_size);
    vector<FieldElement> domain(poly_size);
    
    auto setup_start = chrono::high_resolution_clock::now();
    
    FieldElement generator = get_primitive_nth_root(poly_size);
    
    FieldElement cur(1);
    
    for(int i = 0; i < poly_size; i++) {
        coeffs[i] = FieldElement(i % 1000);  // 반복되는 값으로 설정
        domain[i] = cur;
        cur = cur * generator;
    }
    
    Polynomial poly(coeffs);
    
    auto setup_end = chrono::high_resolution_clock::now();
    auto setup_duration = chrono::duration_cast<chrono::milliseconds>(setup_end - setup_start);
    
    cout << "Setup time: " << setup_duration.count() << " ms" << endl << endl;
    
    // ========================================
    // 1. Naive Evaluation (작은 크기로만)
    // ========================================
    cout << "--- Method 1: Naive Evaluation ---" << endl;
    
    int test_size = (1 << 16);  // 테스트용: 2^16 크기
    vector<FieldElement> test_domain(domain.begin(), domain.begin() + test_size);
    vector<FieldElement> test_coeffs(coeffs.begin(), coeffs.begin() + test_size);
    Polynomial test_poly(test_coeffs);
    
    auto naive_start = chrono::high_resolution_clock::now();
    vector<FieldElement> naive_evals = eval_domain_naive(test_poly, test_domain);
    auto naive_end = chrono::high_resolution_clock::now();
    
    auto naive_duration = chrono::duration_cast<chrono::milliseconds>(naive_end - naive_start);
    cout << "Time for 2^16 points: " << naive_duration.count() << " ms" << endl;
    cout << "First 5 evaluations: ";
    for(int i = 0; i < 5; i++) {
        cout << naive_evals[i] << " ";
    }
    cout << endl << endl;
    
    // ========================================
    // 2. NTT-based Evaluation (전체 크기)
    // ========================================
    cout << "--- Method 2: NTT-based Evaluation ---" << endl;
    
    auto ntt_start = chrono::high_resolution_clock::now();
    vector<FieldElement> ntt_evals = eval_domain_ntt(poly, domain);
    auto ntt_end = chrono::high_resolution_clock::now();
    
    auto ntt_duration = chrono::duration_cast<chrono::milliseconds>(ntt_end - ntt_start);
    cout << "Time for 2^" << log_size << " points: " << ntt_duration.count() << " ms" << endl;
    cout << "First 5 evaluations: ";
    for(int i = 0; i < 5; i++) {
        cout << ntt_evals[i] << " ";
    }
    cout << endl << endl;
    
    // ========================================
    // 3. FRI Commitment
    // ========================================
    cout << "--- FRI Commitment Generation ---" << endl;
    
    auto fri_start = chrono::high_resolution_clock::now();
    FRICommitment fri = fri_commitment(poly, domain, log_size);
    auto fri_end = chrono::high_resolution_clock::now();
    
    auto fri_duration = chrono::duration_cast<chrono::milliseconds>(fri_end - fri_start);
    cout << "\nFRI commitment time: " << fri_duration.count() << " ms" << endl;
    cout << "Total layers: " << fri.layer_hashes.size() << endl;
    cout << "Final root: " << fri.layer_hashes.back().substr(0, 32) << "..." << endl << endl;
    
    // ========================================
    // 4. Performance Summary
    // ========================================
    auto total_end = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(total_end - total_start);
    
    cout << "=== Performance Summary ===" << endl;
    cout << "Naive (2^16):  " << setw(6) << naive_duration.count() << " ms  " 
         << "(O(n^2))" << endl;
    cout << "NTT (2^20):    " << setw(6) << ntt_duration.count() << " ms  " 
         << "(O(n log n))" << endl;
    if(ntt_duration.count() > 0) {
        double speedup = (double)naive_duration.count() / ntt_duration.count();
        cout << "Speed-up (actual): " << fixed << setprecision(2) << speedup << "x (2^16 naive vs 2^20 NTT)" << endl;
        cout << "Theoretical (size-adjusted): ~200x" << endl;
    }
    cout << "FRI (all):     " << setw(6) << fri_duration.count() << " ms" << endl;
    cout << "Total time:    " << total_duration.count() << " ms" << endl;
    
    // ========================================
    // 5. Verify NTT correctness (작은 크기)
    // ========================================
    cout << endl << "=== NTT Correctness Check ===" << endl;
    int verify_size = (1 << 10);  // 2^10
    vector<FieldElement> verify_coeffs(verify_size);
    vector<FieldElement> verify_domain(verify_size);
    
    FieldElement verify_generator = get_primitive_nth_root(verify_size);  // ← 2^10용!
    FieldElement verify_cur(1);
    for(int i = 0; i < verify_size; i++) {
        verify_domain[i] = verify_cur;
        verify_cur = verify_cur * verify_generator;  // ← 올바른 크기의 generator
    }

    
    Polynomial verify_poly(verify_coeffs);
    
    // 방법 1: Naive
    auto verify_naive = eval_domain_naive(verify_poly, verify_domain);
    
    // 방법 2: NTT
    auto verify_ntt = eval_domain_ntt(verify_poly, verify_domain);
    
    // 비교
    int mismatches = 0;
    for(int i = 0; i < verify_size; i++) {
        if(verify_naive[i] != verify_ntt[i]) {
            mismatches++;
        }
    }
    
    if(mismatches == 0) {
        cout << "✓ NTT results match naive evaluation (2^10 test)" << endl;
    } else {
        cout << "✗ Mismatch detected: " << mismatches << " / " << verify_size << endl;
    }
    
    return 0;
}
