#include <iostream>
#include <vector>
#include "polynomial.h"
#include "merkle.h"
#include <chrono>
#include <string>
using namespace std;

vector<FieldElement> eval_domain(Polynomial poly, vector<FieldElement> domain) {
    vector<FieldElement> res;
    for(int i = 0; i < domain.size(); i++) {
        res.push_back(poly(domain[i]));
    }
    return res;
}

string fri_commitment(Polynomial poly, vector<FieldElement> domain, int len) {
    // domain에 제곱을 해야 한다.
    
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // poly의 차수는 2^20 -1 --> 상수항 포함하면 미지수 개수는 1 << 20
    int poly_size = (1 << 20);
    vector<FieldElement> coeffs;
    vector<FieldElement> domain;
    FieldElement generator = FieldElement((int128)944590864);


    //cout << generator << endl;
    coeffs.resize((1 << 20));
    domain.resize((1 << 20));
    FieldElement cur = FieldElement(1);
    for(int i = 0; i < (1 << 20); i++) {
        coeffs[i] = FieldElement(i);
        domain[i] = cur;
        cur = cur * generator;
    }
    
    // polynomial을 만든다.
    Polynomial poly(coeffs);
    int times = 0;
    for(int len = (1 << 20); len > 0; len = len >> 1) {
        times++;
    }
    cout << times << endl;
    auto end = chrono::high_resolution_clock::now();

    auto duration_ms = chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << "Time: " << duration_ms.count() << " ms" << endl;

    return 0;
}