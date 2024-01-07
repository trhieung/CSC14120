#include "./final.cuh"

#include <cuda_runtime.h>

// Kernel function to perform im2col on GPU
__global__ void im2col_kernel(const float* image, float* data_col,
                            int height_in, int width_in,
                            int height_kernel, int width_kernel,
                            int height_out, int width_out,
                            int channel_in, int stride, int pad_h, int pad_w) {
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

// Wrapper function for GPU im2col
void im2col_gpu(const float* image, float* data_col,
                int height_in, int width_in,
                int height_kernel, int width_kernel,
                int height_out, int width_out,
                int channel_in, int stride, int pad_h, int pad_w) {
    int block_size = 256;
    int grid_size = (height_out * channel_in * height_kernel * width_out + block_size - 1) / block_size;

    im2col_kernel<<<grid_size, block_size>>>(image, data_col,
                                            height_in, width_in,
                                            height_kernel, width_kernel,
                                            height_out, width_out,
                                            channel_in, stride, pad_h, pad_w);

    cudaDeviceSynchronize();
}
