# 1. 컴파일러 설정
CXX = g++

# 2. 컴파일 옵션 설정
# -std=c++17: C++17 표준 사용 (필요에 따라 c++20 등으로 변경)
# -O3: 최대 최적화 (NTT 성능을 위해 필수)
# -march=native: 현재 CPU 아키텍처에 맞춰 최적화
# -Wall: 모든 경고 메시지 출력
# -fopenmp: OpenMP를 사용 중이라면 주석을 해제하세요
CXXFLAGS = -std=c++17 -O3 -Wall -march=native # -fopenmp

# 3. 실행 파일 이름
TARGET = fri

# 4. 소스 파일 및 오브젝트 파일 리스트
SRCS = fri_ntt_main.cpp polynomial.cpp
OBJS = $(SRCS:.cpp=.o)

# 5. 기본 타겟 (make 입력 시 실행됨)
all: $(TARGET)

# 6. 링크 규칙 (오브젝트 파일들을 모아 실행 파일 생성)
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 7. 컴파일 규칙 (.cpp 파일을 .o 파일로 변환)
# 소스 코드가 변경된 파일만 다시 컴파일합니다.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 8. 정리 규칙 (생성된 파일 삭제)
clean:
	rm -f $(OBJS) $(TARGET)
