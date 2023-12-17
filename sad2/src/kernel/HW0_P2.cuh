#ifndef HW0_P2_H
#define HW0_P2_H

#include <stdio.h>
#include <math.h>
#include"./Check.cuh"
#include"./GpuTimer.cuh"

void addVecOnHost(float* in1, float* in2, float* out, int n);

__global__ void addVecOnDeviceVersion1(float* in1, float* in2, float* out, int n); // Each thread block processes 2 * blockDim.x consecutive elements that form two sections
__global__ void addVecOnDeviceVersion2(float* in1, float* in2, float* out, int n) // use each thread to calculate two adjacent elements of a vector addition.

void addVec(float* in1, float* in2, float* out, int n, int ver=0);
void evaluateTime(int n);

void HW0_P2();

#endif // HW0_H
