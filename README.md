# Introduction

zk-stark를 구현하기 위해 필요한 fft 알고리즘을 cuda로 구현한 코드입니다.

https://cp-algorithms.com/algebra/fft.html 해당 알고리즘을 사용하였고, gpu에서는 cooley-turkey 알고리즘을 사용하였습니다.

## 구성 파일

fft.cu: global memory만 사용해서 fft 구현

fft_optimized.cu: global memory와 shared memory를 사용해서 fft 구현

## 성능 비교

fft.cu: 평균 251.18ms

fft_optimized.cu: 평균 2.17ms
