# ZK-STARK FRI 프로토콜의 GPU/CPU 가속 및 메모리 계층 병목 분석

## 📖 Introduction

[cite_start]ZK-STARK는 양자 컴퓨팅으로부터도 안전하고 trusted setup이 필요 없는 영지식 증명(Zero-Knowledge Proof) 기술로 주목받고 있습니다[cite: 5]. [cite_start]하지만 방대한 연산량과 메모리 접근을 요구하여, 증명 생성 시간이 서비스 적용의 가장 큰 걸림돌이 되고 있습니다[cite: 6].

[cite_start]본 프로젝트는 ZK-STARK의 핵심 구성 요소인 FRI 프로토콜이 대규모 다항식 연산과 머클 트리(Merkle Tree) 구성을 포함하여 시스템의 연산 능력(computer power)과 메모리 대역폭(memory bandwidth)을 동시에 강하게 요구한다는 점에 착안하였습니다[cite: 7]. [cite_start]다항식의 차수가 커질수록 시간이 기하급수적으로 증가하는 기존 단일 스레드 CPU 환경의 한계를 극복하기 위해, GPU와 데이터 병렬 처리를 활용한 가속 성능을 비교 분석합니다[cite: 9, 13].

## 🎯 Objectives

- [cite_start]다항식의 규모에 따른 FRI 프로토콜을 **Baseline (Single thread)**, **OpenMP (Multi-thread)**, **CUDA** 세 가지 환경에서 구현하고 각각의 성능을 정량적으로 분석합니다[cite: 15, 16].
- [cite_start]Product 수준으로 사용할 수 있도록 Fiat-Shamir 변환을 적용한 Non-interactive 증명 시스템을 제공합니다[cite: 17].

## ⚙️ Algorithms & Optimizations

[cite_start]NTT 및 머클 트리 연산 속도를 극대화하기 위해 다음과 같은 최적화 포인트를 적용했습니다[cite: 52].

- [cite_start]**CPU-GPU 데이터 통신 최소화**: 다항식의 계수와 x값들을 초기 1회만 GPU로 전송하고, 머클 트리 생성이 완료된 후에만 CPU로 반환하도록 설계하여 데이터 이동 비용을 최소화했습니다[cite: 53, 54, 55].
- [cite_start]**Domain Generation 최적화**: FFT의 x값 요소들을 GPU로 생성할 때 단순히 여러 번 곱하는 대신, 비트(bit) 표현을 활용하여 $\log n$ 번의 연산만으로 처리하도록 속도를 향상시켰습니다[cite: 57, 58, 60].
- [cite_start]**메모리 계층 활용 (Global & Shared Memory)**: In-place 알고리즘의 특성상 발생하는 Global Memory 접근 병목을 해소하기 위해, Shared Memory를 사용하는 단계와 Global Memory를 사용하는 단계를 분리하여 성능을 크게 향상시켰습니다[cite: 62, 63].

## 💻 Environment

- [cite_start]**Server**: 성균관대학교 인의예지 서버 (ji 서버) [cite: 66]
- [cite_start]**GPU**: NVIDIA TITAN V [cite: 66]

## 📊 Results

[cite_start]가장 큰 입력 크기인 $N=2^{24}$ 다항식을 기준으로 측정한 성능 차이는 다음과 같습니다[cite: 113].

### 1. NTT (Number Theoretic Transform)

- [cite_start]**CPU (Single Thread)**: 약 18365ms (18.3초) 소요 [cite: 114]
- [cite_start]**CPU (OpenMP)**: 약 2704ms (2.7초) 소요 - Single 대비 약 6.8배 향상 [cite: 115]
- [cite_start]**GPU (CUDA)**: 약 22.48ms (0.02초) 소요 - **Single 대비 약 816배 향상** [cite: 116]

### 2. FRI Commitment

- [cite_start]**CPU (Single Thread)**: 약 227556ms (227초) 소요 [cite: 120]
- [cite_start]**CPU (OpenMP)**: 약 141344ms (141초) 소요 - Single 대비 약 1.6배 향상 [cite: 121]
- [cite_start]**GPU (CUDA)**: 약 560.35ms (0.56초) 소요 - **Single 대비 약 406배 향상** [cite: 122]

[cite_start]💡 **성능 분석**: FRI commitment 단계는 대량의 해시 연산과 빈번한 메모리 접근을 요구하여 CPU 환경에서는 메모리 대역폭이 강력한 병목으로 작용합니다[cite: 123, 124]. [cite_start]하지만 GPU의 병렬 아키텍처를 통해 이러한 병목을 효과적으로 해소할 수 있었습니다[cite: 125].

## 📝 Conclusion

[cite_start]입력 크기가 $2^{24}$에 달하는 대규모 연산 환경에서 CPU 기반 처리는 분 단위의 시간이 소요되어 실시간 증명 생성에 부적합함을 확인하였습니다[cite: 128]. [cite_start]반면, CUDA 기반 GPU 가속을 적용할 경우 연산 시간을 ms 단위로 획기적으로 단축할 수 있었습니다[cite: 129]. [cite_start]이더리움의 블록 생성 시간이 15초인 것을 감안한다면, 블록 생성 시간 내에 안정적으로 증명을 생성하기 위해 GPU 가속을 사용하는 것은 선택이 아닌 필수입니다[cite: 130].

## 📚 References

- [1] zk-stark: Eli Ben-Sasson et al. [cite_start]Scalable, transparent, and post-quantum secure computational integrity (2018)[cite: 134, 135].
- [2] Fri-protocol: Eli Ben-Sasson et al. [cite_start]Fast Reed-Solomon Interactive Oracle Proofs of Proximity (2018)[cite: 136, 137].
- [cite_start][3] CP-Algorithms: FFT[cite: 138].
- [cite_start][4] Wikipedia: Merkle tree[cite: 139].
