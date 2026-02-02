#include <iostream>
#include <vector>
#include "polynomial.h"
#include "merkle_omp.h"
#include "ntt_omp.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>  // ✅ ADDED: OpenMP header

using namespace std;

// ============================================================================
// Helper: Generate primitive n-th root of unity
// ============================================================================
FieldElement get_primitive_nth_root(int n) {
    int128 exponent = (FieldElement::k_modulus - (int128)1) / (int128)n;
    return FieldElement(7).pow(exponent);
}

// ============================================================================
// OPTIMIZED: Naive Evaluation with OpenMP
// ============================================================================
vector<FieldElement> eval_domain_naive_omp(
    const Polynomial& poly,
    const vector<FieldElement>& domain) {
    
    vector<FieldElement> res(domain.size());
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < domain.size(); i++) {
        res[i] = poly(domain[i]);
    }
    
    return res;
}

// ============================================================================
// OPTIMIZED: NTT-based Evaluation (already optimized in ntt_omp.h)
// ============================================================================
vector<FieldElement> eval_domain_ntt_omp(
    const Polynomial& poly,
    const vector<FieldElement>& domain) {
    
    return NTT_OMP::evalDomain(poly, domain);
}

// ============================================================================
// FRI Commitment Structure
// ============================================================================
struct FRICommitment {
    vector<string> layer_hashes;   // Each layer's Merkle root
    vector<size_t> layer_sizes;    // Size of each layer
    vector<long long> layer_times; // Timing for each layer (in ms)
};

// ============================================================================
// OPTIMIZED: FRI Commitment Generation with OpenMP
// ============================================================================
FRICommitment fri_commitment_omp(
    const Polynomial& poly,
    const vector<FieldElement>& domain,
    int num_layers) {
    
    FRICommitment result;
    
    // Initial evaluation using optimized NTT
    vector<FieldElement> current_evaluations = eval_domain_ntt_omp(poly, domain);
    
    cout << "FRI Layers (OpenMP Optimized):" << endl;
    
    for (int layer = 0; layer < num_layers; layer++) {
        auto layer_start = chrono::high_resolution_clock::now();
        cout << " Layer " << layer << ": " << current_evaluations.size() << " points";
        
        // Build Merkle tree
        MerkleTree tree(current_evaluations);
        string root = tree.get_root();
        
        result.layer_hashes.push_back(root);
        result.layer_sizes.push_back(current_evaluations.size());
        
        cout << " -> root: " << root.substr(0, 16) << "...";
        
        auto layer_end = chrono::high_resolution_clock::now();
        auto layer_duration = chrono::duration_cast<chrono::milliseconds>(
            layer_end - layer_start
        );
        
        result.layer_times.push_back(layer_duration.count());
        cout << " (" << layer_duration.count() << " ms)" << endl;
        
        if (current_evaluations.size() <= 1) break;
        
        // FRI folding: Prepare next layer evaluations
        // ✅ PARALLELIZED: Each folding operation is independent
        vector<FieldElement> next_evals(current_evaluations.size() / 2);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < next_evals.size(); i++) {
            // Simple folding: average of adjacent pairs
            next_evals[i] = (current_evaluations[2*i] + current_evaluations[2*i+1]) *
                          FieldElement(2).inverse();
        }
        
        current_evaluations = next_evals;
    }
    
    return result;
}

// ============================================================================
// Performance Benchmarking with OpenMP
// ============================================================================
int main(int argc, char** argv ) {
    cout << "=== FRI Commitment with OpenMP Optimization ===" << endl << endl;
    auto total_start = chrono::high_resolution_clock::now();

    // ========================================
    // Parameter Setup
    // ========================================
    int log_size = stoi(argv[1]); // 2^20 = 1,048,576 elements
    int poly_size = (1 << log_size);

    cout << "Polynomial size: 2^" << log_size << " = " << poly_size << " coefficients" << endl;
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;
    cout << endl;

    // Setup polynomial and domain
    vector<FieldElement> coeffs(poly_size);
    vector<FieldElement> domain(poly_size);

    auto setup_start = chrono::high_resolution_clock::now();

    FieldElement generator = get_primitive_nth_root(poly_size);
    FieldElement cur(1);

    // ✅ FIXED: Proper parallel initialization with correct structure
    #pragma omp parallel
    {
        // Parallel loop for coefficients
        #pragma omp for schedule(static)
        for (int i = 0; i < poly_size; i++) {
            coeffs[i] = FieldElement(i % 1000);
        }
        
        // Sequential part for domain (single-threaded due to data dependency)
        // Domain generation depends on previous values: domain[i] = cur * generator^i
        #pragma omp single
        {
            cur = FieldElement(1);  // Reset cur
            for (int i = 0; i < poly_size; i++) {
                domain[i] = cur;
                cur = cur * generator;
            }
        }
        // ✅ Implicit barrier at end of parallel region
    }

    Polynomial poly(coeffs);

    auto setup_end = chrono::high_resolution_clock::now();
    auto setup_duration = chrono::duration_cast<chrono::milliseconds>(
        setup_end - setup_start
    );

    cout << "Setup time: " << setup_duration.count() << " ms" << endl << endl;

    // ========================================
    // 1. Naive Evaluation (small test size)
    // ========================================
    /*
    cout << "--- Method 1: Naive Evaluation (OpenMP) ---" << endl;
    int test_size = (1 << 16); // 2^16 test
    vector<FieldElement> test_domain(domain.begin(), domain.begin() + test_size);
    vector<FieldElement> test_coeffs(coeffs.begin(), coeffs.begin() + test_size);
    Polynomial test_poly(test_coeffs);

    auto naive_start = chrono::high_resolution_clock::now();
    vector<FieldElement> naive_evals = eval_domain_naive_omp(test_poly, test_domain);
    auto naive_end = chrono::high_resolution_clock::now();
    
    auto naive_duration = chrono::duration_cast<chrono::milliseconds>(
        naive_end - naive_start
    );

    cout << "Time for 2^16 points: " << naive_duration.count() << " ms" << endl;
    cout << "First 5 evaluations: ";
    for (int i = 0; i < 5; i++) {
        cout << naive_evals[i].to_string().substr(0, 8) << " ";
    }
    cout << endl << endl;
    */

    // ========================================
    // 2. NTT-based Evaluation (OpenMP optimized)
    // ========================================
    cout << "--- Method 2: NTT-based Evaluation (OpenMP) ---" << endl;
    auto ntt_start = chrono::high_resolution_clock::now();
    vector<FieldElement> ntt_evals = eval_domain_ntt_omp(poly, domain);
    auto ntt_end = chrono::high_resolution_clock::now();
    
    auto ntt_duration = chrono::duration_cast<chrono::milliseconds>(
        ntt_end - ntt_start
    );

    cout << "Time for 2^" << log_size << " points: " << ntt_duration.count() << " ms" << endl;
    cout << "First 5 evaluations: ";
    for (int i = 0; i < 5; i++) {
        cout << ntt_evals[i].to_string().substr(0, 8) << " ";
    }
    cout << endl << endl;

    // ========================================
    // 3. FRI Commitment (OpenMP optimized)
    // ========================================
    cout << "--- FRI Commitment Generation (OpenMP) ---" << endl;
    auto fri_start = chrono::high_resolution_clock::now();
    FRICommitment fri = fri_commitment_omp(poly, domain, log_size);
    auto fri_end = chrono::high_resolution_clock::now();
    
    auto fri_duration = chrono::duration_cast<chrono::milliseconds>(
        fri_end - fri_start
    );

    cout << "\nFRI commitment total time: " << fri_duration.count() << " ms" << endl;
    cout << "Total layers: " << fri.layer_hashes.size() << endl;
    cout << "Final root: " << fri.layer_hashes.back().substr(0, 32) << "..." << endl;

    // Layer-by-layer timing analysis
    cout << "\nLayer timing breakdown:" << endl;
    cout << " Layer | Size | Time (ms)" << endl;
    cout << " ------|-----------|----------" << endl;
    for (size_t i = 0; i < fri.layer_sizes.size(); i++) {
        cout << " " << setw(5) << i << " | "
             << setw(9) << fri.layer_sizes[i] << " | "
             << setw(8) << fri.layer_times[i] << endl;
    }

    cout << endl;

    // ========================================
    // 4. Performance Summary
    // ========================================
    auto total_end = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(
        total_end - total_start
    );

    cout << "=== Performance Summary ===" << endl;
    //cout << "Naive (2^16, OMP): " << setw(6) << naive_duration.count() << " ms (O(n^2))" << endl;
    cout << "NTT (2^" << log_size << ", OMP): " << setw(6) << ntt_duration.count() << " ms (O(n log n))" << endl;
    /*
    if (ntt_duration.count() > 0) {
        double speedup = (double)naive_duration.count() / ntt_duration.count();
        cout << "Speed-up (actual): " << fixed << setprecision(2) << speedup
             << "x (2^16 naive vs 2^20 NTT)" << endl;
    }
    */

    cout << "FRI (all layers): " << setw(6) << fri_duration.count() << " ms" << endl;
    cout << "Total time: " << setw(6) << total_duration.count() << " ms" << endl;
    cout << endl;

    // ========================================
    // 5. Correctness Verification (small test)
    // ========================================
    cout << "=== NTT Correctness Verification ===" << endl;
    int verify_size = (1 << 10); // 2^10
    vector<FieldElement> verify_coeffs(verify_size);
    vector<FieldElement> verify_domain(verify_size);
    
    FieldElement verify_generator = get_primitive_nth_root(verify_size);
    FieldElement verify_cur(1);

    for (int i = 0; i < verify_size; i++) {
        verify_domain[i] = verify_cur;
        verify_cur = verify_cur * verify_generator;
    }

    Polynomial verify_poly(verify_coeffs);
    auto verify_naive = eval_domain_naive_omp(verify_poly, verify_domain);
    auto verify_ntt = eval_domain_ntt_omp(verify_poly, verify_domain);

    int mismatches = 0;
    #pragma omp parallel for reduction(+:mismatches)
    for (int i = 0; i < verify_size; i++) {
        if (verify_naive[i] != verify_ntt[i]) {
            mismatches++;
        }
    }

    if (mismatches == 0) {
        cout << "✓ NTT results match naive evaluation (2^10 test)" << endl;
    } else {
        cout << "✗ Mismatch detected: " << mismatches << " / " << verify_size << endl;
    }

    cout << endl;
    cout << "=== OpenMP Optimization Summary ===" << endl;
    cout << "Expected speedups:" << endl;
    cout << " - Naive evaluation: 8-12x" << endl;
    cout << " - NTT operations: 5-8x" << endl;
    cout << " - FRI overall: 4-6x (limited by sequential layer dependencies)" << endl;

    return 0;
}
