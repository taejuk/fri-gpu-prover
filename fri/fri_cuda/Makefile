NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70
HDRS      = poseidon.cuh poseidon_constants.h

# 빌드할 실행파일 목록 — 여기에 추가만 하면 됨
TARGETS = main

all: poseidon_constants.h $(TARGETS)

# 상수 헤더 자동 생성
poseidon_constants.h: gen_poseidon_constants.py
	python3 $< $@

# 패턴 규칙: <name> → <name>.cu
# fri_cuda → fri_cuda.cu
# fri_cuda_pinned → fri_cuda_pinned.cu
%: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ===================== FRI RPC Server (GPU + RPC) =====================
RPC_X      = fri_rpc.x
RPC_HDR    = fri_rpc.h
RPC_SVC    = fri_rpc_svc.c
RPC_CLNT   = fri_rpc_clnt.c
RPC_XDR    = fri_rpc_xdr.c
RPC_SERVER = fri_rpc_server.c

# libtirpc (rpc/rpc.h) 경로와 라이브러리
RPC_CFLAGS = $(shell pkg-config --cflags libtirpc 2>/dev/null || echo "-I/usr/include/tirpc")
RPC_LIBS   = -ltirpc -lpthread -lcudart

fri_rpc_server: $(RPC_SERVER) $(RPC_SVC) $(RPC_XDR) $(RPC_HDR) poseidon_constants.h
	$(NVCC) $(NVCCFLAGS) $(RPC_CFLAGS) -o $@ \
		$(RPC_SERVER) $(RPC_SVC) $(RPC_XDR) \
		$(RPC_LIBS)

fri_rpc_client: fri_rpc_client.c $(RPC_CLNT) $(RPC_XDR)
	gcc $(RPC_CFLAGS) -o $@ fri_rpc_client.c $(RPC_CLNT) $(RPC_XDR) -ltirpc -lnsl

# rpcgen 자동 생성
$(RPC_HDR) $(RPC_SVC) $(RPC_CLNT) $(RPC_XDR) $(RPC_SERVER): $(RPC_X)
	rpcgen $(RPC_X)

clean:
	rm -f fri_rpc_server $(RPC_HDR) $(RPC_SVC) $(RPC_CLNT) $(RPC_XDR) *.o poseidon_constants.h $(TARGETS) fri_rpc_client.c
