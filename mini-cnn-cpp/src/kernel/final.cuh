#ifndef FINAL_H
#define FINAL_H

#include <stdio.h>
#include <math.h>
#include <iostream>
#include"./Check.cuh"
#include"./GpuTimer.cuh"

// Kernel function to perform im2col on GPU
__global__ void im2col_kernel(float* image, float* data_col,
                            int height_in, int width_in,
                            int height_kernel, int width_kernel,
                            int height_out, int width_out,
                            int channel_in, int stride, int pad_h, int pad_w);

// Wrapper function to call im2col_kernel from CPU
void im2col_gpu(const float* image, float* data_col,
                int height_in, int width_in,
                int height_kernel, int width_kernel,
                int height_out, int width_out,
                int channel_in, int stride, int pad_h, int pad_w);

#endif // FINAL_H