# 1. 컴파일러 설정
CXX = g++

# 2. 컴파일 옵션
# -std=c++17: 최신 C++ 표준 사용
# -O3: 최대 최적화 (NTT 성능 필수)
# -fopenmp: OpenMP 병렬 처리 활성화 (필수)
# -march=native: 현재 CPU 아키텍처에 맞게 최적화 (AVX 등 활용)
# -Wall: 경고 메시지 출력
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -Wall

# 3. 링커 옵션
# 링크 단계에서도 OpenMP 라이브러리를 연결해야 합니다.
LDFLAGS = -fopenmp

# 4. 생성할 실행 파일 이름
TARGET = fri_ntt_omp

# 5. 소스 파일 목록 (.cpp 파일만 나열)
SRCS = fri_ntt_main_omp.cpp polynomial.cpp

# 6. 오브젝트 파일 목록 (소스 파일 이름에서 .cpp를 .o로 치환)
OBJS = $(SRCS:.cpp=.o)

# 7. 기본 타겟
all: $(TARGET)

# 8. 링크 규칙: 오브젝트 파일들을 모아 실행 파일 생성
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# 9. 컴파일 규칙: .cpp 파일을 .o 파일로 컴파일
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 10. 정리 규칙
clean:
	rm -f $(OBJS) $(TARGET)

# (선택 사항) 헤더 파일 의존성 명시
# 헤더 파일이 바뀌었을 때 관련 cpp가 다시 컴파일되도록 강제하고 싶다면 아래 주석을 참고하세요.
fri_ntt_main_omp.o: fri_ntt_main_omp.cpp field.h merkle.h merkle_omp.h ntt_omp.h polynomial.h
polynomial.o: polynomial.cpp polynomial.h field.h
