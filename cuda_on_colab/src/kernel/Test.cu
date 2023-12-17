#include "./cuda_file.cuh"
#include <cstdio>

__global__ void cudaKernel() {
    printf("Hello from CUDA!\n");
}

void runCudaFunction() {
    cudaKernel<<<2, 2>>>();
    cudaDeviceSynchronize();
}
