#include "./cuda_file.cuh"

__global__ void cudaKernel() {
    printf("Hello from CUDA!\n");
}

void runCudaFunction() {
    cudaKernel<<<2, 2>>>();
    CHECK(cudaDeviceSynchronize());
}
