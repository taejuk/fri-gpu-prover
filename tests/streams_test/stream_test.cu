#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.1415,i));
    }
}

int main() {
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);

	cudaMalloc(&data[i], N * sizeof(float));

	kernel<<<1,64,0,streams[i]>>>(data[i], N);
        //kernel<<<1,64>>>(data[i], N);
	kernel<<<1,1>>>(0,0);
    
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time: " << milliseconds << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();

    return 0;
}
