NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70
HDRS      = poseidon.cuh poseidon_constants.h

# 빌드할 실행파일 목록 — 여기에 추가만 하면 됨
TARGETS = fri_cuda fri_cuda_pinned

all: poseidon_constants.h $(TARGETS)

# 상수 헤더 자동 생성
poseidon_constants.h: gen_poseidon_constants.py
	python3 $< $@

# 패턴 규칙: <name> → <name>.cu
# fri_cuda → fri_cuda.cu
# fri_cuda_pinned → fri_cuda_pinned.cu
%: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) poseidon_constants.h
