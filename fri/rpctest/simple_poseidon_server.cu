#include "simple_poseidon.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../fri_cuda/util.cuh"
#include "../fri_cuda/poseidon.cuh"

static int cuda_initialized = 0;

static void init_cuda(void) {
    if (cuda_initialized) return;
    cudaError_t err = poseidon_init();
    if (err != cudaSuccess) {
        fprintf(stderr, "poseidon_init failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cuda_initialized = 1;
    printf("✅ CUDA + Poseidon initialized for simple RPC server\n");
}

CompressResult *poseidon_compress_1_svc(CompressInput *argp, struct svc_req *rqstp) {
    static CompressResult result;

    init_cuda();

    uint64_t left  = argp->left;
    uint64_t right = argp->right;

    // GPU에서 Poseidon compress 호출
    uint64_t res = poseidon_compress(left, right);

    result.result = res;
    return &result;
}

/* rpcgen이 만든 main() + svc_run() */
int main(int argc, char **argv) {
    register SVCXPRT *transp;

    pmap_unset(SIMPLE_POSEIDON_PROG, SIMPLE_V1);

    transp = svcudp_create(RPC_ANYSOCK);
    if (transp == NULL) {
        fprintf(stderr, "cannot create udp service.\n");
        exit(1);
    }
    svc_register(transp, SIMPLE_POSEIDON_PROG, SIMPLE_V1, poseidon_compress_1_svc, IPPROTO_UDP);

    transp = svctcp_create(RPC_ANYSOCK, 0, 0);
    if (transp == NULL) {
        fprintf(stderr, "cannot create tcp service.\n");
        exit(1);
    }
    svc_register(transp, SIMPLE_POSEIDON_PROG, SIMPLE_V1, poseidon_compress_1_svc, IPPROTO_TCP);

    printf("🚀 Simple Poseidon RPC Server started (UDP + TCP)\n");
    printf("   → poseidon_compress(left, right) 서비스 대기 중...\n");
    svc_run();
    fprintf(stderr, "svc_run returned\n");
    exit(1);
}
