#!/usr/bin/env python3
"""
BabyBear Poseidon12 CPU 레퍼런스 구현 + 테스트 벡터 생성
=========================================================
GPU 구현(poseidon.cuh)과 bit-exact 일치 확인에 사용.

사용법:
  python3 verify_poseidon.py              # 테스트 벡터 출력
  python3 verify_poseidon.py --vectors    # GPU 코드에 삽입할 C 배열 출력
"""

import hashlib
import sys

# ── 파라미터 ──────────────────────────────────────────────────────────────

P     = 2013265921
T     = 12
ALPHA = 7
RF    = 8
RP    = 22
TOTAL_ROUNDS = RF + RP   # 30
TOTAL_RC     = TOTAL_ROUNDS * T  # 360


# ── 상수 (gen_poseidon_constants.py 와 동일 로직) ─────────────────────────

def mod_inv(a, p=P):
    return pow(a, p - 2, p)

def _gen_rc():
    domain = b"Poseidon_BabyBear_p2013265921_t12_a7_RF8_RP22_RC_v1"
    out = []
    idx = 0
    while len(out) < TOTAL_RC:
        d = hashlib.sha256(domain + idx.to_bytes(4, "little")).digest()
        for i in range(0, 32, 4):
            out.append(int.from_bytes(d[i:i+4], "little") % P)
            if len(out) >= TOTAL_RC:
                break
        idx += 1
    return out[:TOTAL_RC]

def _gen_mds():
    x = list(range(T))
    y = list(range(T, 2 * T))
    return [[mod_inv((x[i] - y[j]) % P) for j in range(T)] for i in range(T)]

RC  = _gen_rc()
MDS = _gen_mds()


# ── CPU Poseidon 구현 ─────────────────────────────────────────────────────

def pow7(x):
    x2 = x * x % P
    x4 = x2 * x2 % P
    return x * x2 % P * x4 % P

def mds_mul(state):
    return [sum(MDS[i][j] * state[j] for j in range(T)) % P for i in range(T)]

def poseidon_permutation(state):
    """Poseidon12 퍼뮤테이션 (Python reference)"""
    state = list(state)
    rc = 0

    # 앞쪽 full rounds
    for _ in range(RF // 2):
        state = [(state[i] + RC[rc + i]) % P for i in range(T)]
        rc += T
        state = [pow7(x) for x in state]
        state = mds_mul(state)

    # Partial rounds
    for _ in range(RP):
        state = [(state[i] + RC[rc + i]) % P for i in range(T)]
        rc += T
        state[0] = pow7(state[0])
        state = mds_mul(state)

    # 뒤쪽 full rounds
    for _ in range(RF // 2):
        state = [(state[i] + RC[rc + i]) % P for i in range(T)]
        rc += T
        state = [pow7(x) for x in state]
        state = mds_mul(state)

    assert rc == TOTAL_RC
    return state

def poseidon_compress(left, right):
    """2-to-1 압축: [left, right, 0, ..., 0] → permute → state[0]"""
    state = [0] * T
    state[0] = left
    state[1] = right
    return poseidon_permutation(state)[0]


# ── 테스트 벡터 ───────────────────────────────────────────────────────────

TEST_CASES = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (123456789, 987654321),
    (P - 1, P - 1),
    (1000000000, 500000000),
]

def run_tests():
    print("=" * 60)
    print("BabyBear Poseidon12 CPU 레퍼런스 테스트 벡터")
    print(f"p={P}, t={T}, alpha={ALPHA}, RF={RF}, RP={RP}")
    print("=" * 60)
    print()
    for left, right in TEST_CASES:
        result = poseidon_compress(left, right)
        print(f"compress({left}, {right})")
        print(f"  → {result}")
    print()

    # 퍼뮤테이션 테스트 (state = [0, 1, 2, ..., 11])
    init_state = list(range(T))
    out = poseidon_permutation(init_state)
    print(f"permutation([0,1,...,11])")
    print(f"  → {out}")
    print()
    print("GPU 결과와 위 값이 일치하면 구현 정확.")


def print_c_vectors():
    """GPU 검증 코드에 붙여넣을 C 배열 출력"""
    print("// ── poseidon 테스트 벡터 (poseidon.cuh 검증용) ──")
    print(f"// poseidon_compress(left, right) 결과")
    print("static const struct {")
    print("    uint64_t left, right, expected;")
    print("} POSEIDON_TEST_VECTORS[] = {")
    for left, right in TEST_CASES:
        result = poseidon_compress(left, right)
        print(f"    {{{left}ULL, {right}ULL, {result}ULL}},")
    print("};")
    print(f"static const int N_VECTORS = {len(TEST_CASES)};")
    print()

    # 퍼뮤테이션 벡터
    init_state = list(range(T))
    out = poseidon_permutation(init_state)
    print("// permutation([0,1,...,11]) 결과")
    vals = ", ".join(f"{v}ULL" for v in out)
    print(f"static const uint64_t PERM_EXPECTED[{T}] = {{{vals}}};")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--vectors" in sys.argv:
        print_c_vectors()
    else:
        run_tests()
        print()
        print("C 배열 형태로 보려면:  python3 verify_poseidon.py --vectors")
