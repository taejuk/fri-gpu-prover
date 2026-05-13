#include "fri_rpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <rpc/pmap_clnt.h>
#include <cuda_runtime.h>

#include "util.cuh"
#include "ntt.cuh"
#include "intt.cuh"
#include "fri.cuh"
#include "poseidon.cuh"

static int cuda_initialized = 0;

static void init_cuda(void) {
    if (cuda_initialized) return;
    cudaError_t err = poseidon_init();
    if (err != cudaSuccess) {
        fprintf(stderr, "poseidon_init failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cuda_initialized = 1;
    printf("✅ CUDA + Poseidon12 initialized for RPC server\n");
}

static uint64_t compute_fri_root_gpu(uint64_t* h_coeffs, unsigned int log_n) {
    init_cuda();
    uint64_t p = BABYBEAR_P;
    uint64_t n = 1ULL << log_n;

    uint64_t *d_coeff = NULL, *d_twiddles = NULL, *d_evals = NULL;
    cudaMalloc(&d_coeff, n * sizeof(uint64_t));
    cudaMalloc(&d_twiddles, n * sizeof(uint64_t));
    cudaMalloc(&d_evals, n * sizeof(uint64_t));

    cudaMemcpy(d_coeff, h_coeffs, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_evals, h_coeffs, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t g = get_generator(7, p-1, n, p);
    uint64_t g_inv = mod_inverse_prime(g, p);
    int grid = (n + blockSize - 1) / blockSize;
    generate<<<grid, blockSize>>>(d_twiddles, d_twiddles, g, g_inv, n, p);
    cudaDeviceSynchronize();

    ntt(d_coeff, d_twiddles, n, p);
    cudaMemcpy(d_evals, d_coeff, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    FRICommitmentGPU fri = fri_commitment_gpu(d_evals, n, log_n);

    uint64_t root = 0;
    if (!fri.layer_roots.empty()) {
        sscanf(fri.layer_roots.back().c_str(), "%llx", &root);
    }

    cudaFree(d_coeff);
    cudaFree(d_twiddles);
    cudaFree(d_evals);
    return root;
}

FriResult *compute_fri_root_1_svc(PolyInput *argp, struct svc_req *rqstp) {
    static FriResult result;

    if (argp->log_n < 10 || argp->log_n > 22 ||
        argp->coeffs.coeffs_len != (1U << argp->log_n)) {
        result.root = 0;
        return &result;
    }

    result.root = compute_fri_root_gpu(argp->coeffs.coeffs_val, argp->log_n);
    return &result;
}

/* ── rpcgen -a 가 생성한 main() ── */
int main(int argc, char **argv) {
    register SVCXPRT *transp;
    pmap_unset(FRI_RPC_PROG, FRI_V1);

    transp = svcudp_create(RPC_ANYSOCK);
    if (transp == NULL) {
        fprintf(stderr, "cannot create udp service.\n");
        exit(1);
    }
    if (!svc_register(transp, FRI_RPC_PROG, FRI_V1, compute_fri_root_1_svc, IPPROTO_UDP)) {
        fprintf(stderr, "unable to register (FRI_RPC_PROG, FRI_V1, udp).\n");
        exit(1);
    }

    transp = svctcp_create(RPC_ANYSOCK, 0, 0);
    if (transp == NULL) {
        fprintf(stderr, "cannot create tcp service.\n");
        exit(1);
    }
    if (!svc_register(transp, FRI_RPC_PROG, FRI_V1, compute_fri_root_1_svc, IPPROTO_TCP)) {
        fprintf(stderr, "unable to register (FRI_RPC_PROG, FRI_V1, tcp).\n");
        exit(1);
    }

    printf("🚀 FRI RPC Server started (UDP + TCP)\n");
    svc_run();
    fprintf(stderr, "svc_run returned\n");
    exit(1);
}
