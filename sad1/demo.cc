#include <iostream>
#include <vector>

__global__ void myKernel(int* x, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        printf("hello from kernel, %d\n", x[tid]);
    }
}

int main(void) {
    const int size = 5;
    std::vector<int> x {1, 2, 3, 4, 5};

    int* d_x; // device pointer
    cudaMalloc((void**)&d_x, size * sizeof(int));
    cudaMemcpy(d_x, x.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 2;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, size);
    cudaDeviceSynchronize();

    cudaFree(d_x);

    printf("Hello CUDA!\n");

    return 0;
}
