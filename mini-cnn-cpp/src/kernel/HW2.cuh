#ifndef HW2_H
#define HW2_H

#include <stdio.h>
#include <math.h>
#include"./Check.cuh"
#include"./GpuTimer.cuh"
#define TILE_WIDTH 32

void printDeviceInfo();

// HW2_P1
__global__ void reduceBlksKernel1(int * in, int * out, int n);
__global__ void reduceBlksKernel2(int * in, int * out, int n);
__global__ void reduceBlksKernel3(int * in, int * out, int n);

int reduce(int const * in, int n,
        bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1);

void HW2_P1_checkCorrectness(int r1, int r2);

void HW2_P1();

// HW2_P2
__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k);
__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k);

void matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
    bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1);

float HW2_P2_checkCorrectness(float * a1, float* a2, int n);

void HW2_P2();

#endif // HW2_H
