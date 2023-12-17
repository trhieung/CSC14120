
__global__ void cudaKernel() {
    printf("Hello from CUDA!\n");
}

void runCudaFunction() {
    CHECK(cudaKernel<<<2, 2>>>());
    CHECK(cudaDeviceSynchronize());
}
