# 1. 컴파일러 설정
NVCC = nvcc

# 4. 실행 파일 이름
TARGET = fri_cuda

# 5. 소스 파일 리스트
# 만약 polynomial.cpp도 같이 컴파일해야 한다면 뒤에 추가해주세요.
# 예: SRCS = fri_cuda.cu polynomial.cpp
SRCS = fri_cuda.cu

# 6. 빌드 규칙
all: $(TARGET)
$(TARGET): $(SRCS)
	$(NVCC) -o $@ $^

# 7. 정리 규칙
clean:
	rm -f $(TARGET)
