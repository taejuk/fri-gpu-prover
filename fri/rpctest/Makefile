NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70
RPC_CFLAGS = $(shell pkg-config --cflags libtirpc 2>/dev/null || echo "-I/usr/include/tirpc")

all: simple_poseidon_server simple_poseidon_client

simple_poseidon_server: simple_poseidon_server.cu simple_poseidon_svc.c simple_poseidon_xdr.c
	$(NVCC) $(NVCCFLAGS) $(RPC_CFLAGS) -o $@ $^ -ltirpc -lpthread -lcudart

simple_poseidon_client: simple_poseidon_client.c simple_poseidon_clnt.c simple_poseidon_xdr.c
	gcc $(RPC_CFLAGS) -o $@ $^ -ltirpc -lnsl

clean:
	rm -f simple_poseidon_server simple_poseidon_client *.o
	rm -f simple_poseidon.h simple_poseidon_clnt.c simple_poseidon_svc.c simple_poseidon_xdr.c
