#include "conv_cust1.h"
#include <math.h>
#include "../kernel/HW2.cuh"
#include <iostream>

__global__ void im2col_kernel(const float* image, float* data_col,
                               int height_in, int width_in, int channel_in,
                               int height_kernel, int width_kernel,
                               int height_out, int width_out,
                               int stride, int pad_h, int pad_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < height_out * width_out * channel_in) {
        int c = index / (height_out * width_out);
        int i_out = (index % (height_out * width_out)) / width_out;
        int j_out = (index % (height_out * width_out)) % width_out;

        int start_h = i_out * stride - pad_h;
        int start_w = j_out * stride - pad_w;

        for (int i_kernel = 0; i_kernel < height_kernel; ++i_kernel) {
            for (int j_kernel = 0; j_kernel < width_kernel; ++j_kernel) {
                int h = start_h + i_kernel;
                int w = start_w + j_kernel;

                int image_index = (c * height_in + h) * width_in + w;
                int data_col_index = (i_out * width_out + j_out) * height_kernel * width_kernel * channel_in +
                                     (i_kernel * width_kernel + j_kernel) * channel_in + c;
                
                // Perform vectorized load and store if possible
                data_col[data_col_index] = (h >= 0 && h < height_in && w >= 0 && w < width_in) ?
                                           image[image_index] : 0.0;
            }
        }
    }
}

__global__ void im2col_gpu_kernel(const float* image, float* data_col,
                                   int height_in, int width_in,
                                   int channel_in, int height_kernel, int width_kernel,
                                   int height_out, int width_out,
                                   int stride, int pad_h, int pad_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < height_out * width_out * channel_in) {
        int c = index / (height_out * width_out);
        int i_out = (index % (height_out * width_out)) / width_out;
        int j_out = (index % (height_out * width_out)) % width_out;

        int start_h = i_out * stride - pad_h;
        int start_w = j_out * stride - pad_w;

        for (int i_kernel = 0; i_kernel < height_kernel; ++i_kernel) {
            for (int j_kernel = 0; j_kernel < width_kernel; ++j_kernel) {
                int h = start_h + i_kernel;
                int w = start_w + j_kernel;

                int image_index = (c * height_in + h) * width_in + w;
                int data_col_index = (i_out * width_out + j_out) * height_kernel * width_kernel * channel_in +
                                     (i_kernel * width_kernel + j_kernel) * channel_in + c;
                
                // Perform vectorized load and store if possible
                data_col[data_col_index] = (h >= 0 && h < height_in && w >= 0 && w < width_in) ?
                                           image[image_index] : 0.0;
            }
        }
    }
}

void Conv::im2col_gpu(const Vector& image, Matrix& data_col_gpu) {
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;

    // Allocate GPU memory
    float *image_gpu, *data_col_gpu_ptr;
    cudaMalloc((void**)&image_gpu, sizeof(float) * image.size());
    cudaMalloc((void**)&data_col_gpu_ptr, sizeof(float) * hw_out * hw_kernel * channel_in);

    // Copy input data to GPU
    cudaMemcpy(image_gpu, image.data(), sizeof(float) * image.size(), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);  // You can adjust this value based on your GPU's capabilities
    dim3 blocksPerGrid((hw_out * channel_in + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch GPU kernel
    im2col_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(image_gpu, data_col_gpu_ptr, height_in, width_in,
                                                          channel_in, height_kernel, width_kernel,
                                                          height_out, width_out, stride, pad_h, pad_w);

    // Copy result back to CPU
    data_col_gpu.resize(hw_out, hw_kernel * channel_in);
    cudaMemcpy(data_col_gpu.data(), data_col_gpu_ptr, sizeof(float) * data_col_gpu.size(),
               cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(image_gpu);
    cudaFree(data_col_gpu_ptr);
}


__global__ void matrix_multiplication_kernel_cust1(const float* data_col, const float* weight,
                                                   float* result, int height_out, int width_out,
                                                   int channel_in_hw_kernel, int channel_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height_out && col < channel_out) {
        float value = 0.0f;
        for (int k = 0; k < channel_in_hw_kernel; ++k) {
            value += data_col[row * channel_in_hw_kernel + k] * weight[k * channel_out + col];
        }
        result[row * channel_out + col] = value;
    }
}

void Conv::forward_gpu(const Matrix& bottom) {
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);

    for (int i = 0; i < n_sample; i++) {
        // im2col
        Matrix data_col;
        im2col_gpu(bottom.col(i), data_col);
        data_cols[i] = data_col;

        // conv by product
        GpuTimer timer;
        timer.Start();

        // Allocate GPU memory
        float* _data_col;
        float* _weight;
        float* _result;

        cudaMalloc((void**)&_data_col, sizeof(float) * data_col.size());
        cudaMalloc((void**)&_weight, sizeof(float) * weight.size());
        cudaMalloc((void**)&_result, sizeof(float) * height_out * width_out * channel_out);

        // Copy input data to GPU
        cudaMemcpy(_data_col, data_col.data(), sizeof(float) * data_col.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(_weight, weight.data(), sizeof(float) * weight.size(), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid(channel_out, (height_out * width_out + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch GPU kernels
        im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(_data_col, _weight, height_in, width_in,
                                                          channel_in, height_kernel, width_kernel,
                                                          height_out, width_out, stride, pad_h, pad_w);

        matrix_multiplication_kernel_cust1<<<blocksPerGrid, threadsPerBlock>>>(_data_col, _weight, _result,
                                                                              height_out, width_out,
                                                                              channel_in * height_kernel * width_kernel,
                                                                              channel_out);

        // Copy result back to CPU
        float* result_cpu = new float[height_out * width_out * channel_out];
        cudaMemcpy(result_cpu, _result, sizeof(float) * height_out * width_out * channel_out,
                   cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(_data_col);
        cudaFree(_weight);
        cudaFree(_result);

        timer.Stop();
        float time = timer.Elapsed();
        printf("Processing time (%s): %f ms\n", "use device", time);

        Matrix result = Eigen::Map<Matrix>(result_cpu, channel_out, height_out * width_out);
        result = result.transpose();
        result.rowwise() += bias.transpose();
        top.col(i) = Eigen::Map<Vector>(result.data(), result.size());

        delete[] result_cpu;
    }
}
