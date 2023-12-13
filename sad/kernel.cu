#include "header.cuh"

void __global__ print()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", idx);
}

void f()
{
    print<<<1, 10>>>();
}