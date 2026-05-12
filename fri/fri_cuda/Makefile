NVCC   = nvcc
TARGET = fri_cuda
SRCS   = fri_cuda.cu
HDRS   = poseidon.cuh poseidon_constants.h

NVCCFLAGS = -O2 -arch=sm_70   # TITAN V = sm_70

all: poseidon_constants.h $(TARGET)

# 상수 헤더 — Python 스크립트로 자동 생성
poseidon_constants.h: gen_poseidon_constants.py
	python3 $< $@

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET) poseidon_constants.h
