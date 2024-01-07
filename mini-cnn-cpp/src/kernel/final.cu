#include "./final.cuh"
#include <cuda_runtime.h>

// Kernel function to perform im2col on GPU
__global__ void im2col_kernel(const float* image, float* data_col,
                              int height_in, int width_in,
                              int height_kernel, int width_kernel,
                              int height_out, int width_out,
                              int channel_in, int stride, int pad_h, int pad_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < height_out * width_out * channel_in * height_kernel * width_kernel) {
        int c = tid % channel_in;
        tid /= channel_in;

        int j_out = tid % width_out;
        tid /= width_out;

        int i_out = tid % height_out;
        tid /= height_out;

        int j_kernel = tid % width_kernel;
        int i_kernel = tid / width_kernel;

        int h = i_out * stride - pad_h + i_kernel;
        int w = j_out * stride - pad_w + j_kernel;

        int image_index = (c * height_in + h) * width_in + w;
        int data_col_index = ((i_out * width_out + j_out) * channel_in + c) * height_kernel * width_kernel +
                             i_kernel * width_kernel + j_kernel;

        // Perform vectorized load and store if possible
        data_col[data_col_index] = (h >= 0 && h < height_in && w >= 0 && w < width_in) ?
                                    image[image_index] : 0.0;
    }
}

// Wrapper function for GPU im2col
void im2col_gpu(const float* image, float* data_col,
                int height_in, int width_in,
                int height_kernel, int width_kernel,
                int height_out, int width_out,
                int channel_in, int stride, int pad_h, int pad_w) {
    int block_size = 256;  // You can adjust this based on your device capabilities

    int grid_size = (height_out * width_out * channel_in * height_kernel * width_kernel + block_size - 1) / block_size;

    dim3 block_dim(block_size);
    dim3 grid_dim(grid_size);

    im2col_kernel<<<grid_dim, block_dim>>>(image, data_col,
                                           height_in, width_in,
                                           height_kernel, width_kernel,
                                           height_out, width_out,
                                           channel_in, stride, pad_h, pad_w);

    cudaDeviceSynchronize();
}

// Kernel function to add bias to each element of the result matrix on GPU
__global__ void add_bias_kernel(float* result, const float* bias,
                                int height_out, int width_out, int channel_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < height_out * width_out * channel_out) {
        int c = idx % channel_out;
        idx /= channel_out;

        int j_out = idx % width_out;
        idx /= width_out;

        int i_out = idx % height_out;

        result[(i_out * width_out + j_out) * channel_out + c] += bias[c];
    }
}

// Wrapper function for GPU bias addition
void add_bias_gpu(float* result_gpu, const float* bias, int height_out, int width_out, int channel_out) {
    int block_size = 256;  // You can adjust the block size based on your specific GPU architecture
    int grid_size = (height_out * width_out * channel_out + block_size - 1) / block_size;

    // Allocate GPU memory for bias
    float* bias_gpu;
    cudaMalloc((void**)&bias_gpu, channel_out * sizeof(float));
    cudaMemcpy(bias_gpu, bias, channel_out * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    add_bias_kernel<<<grid_size, block_size>>>(result_gpu, bias_gpu, height_out, width_out, channel_out);

    // Synchronize to make sure the kernel is finished before copying back the result
    cudaDeviceSynchronize();

    // Free GPU memory for bias
    cudaFree(bias_gpu);
}


// #include "./final.cuh"

// #include <cuda_runtime.h>

// // Kernel function to perform im2col on GPU
// __global__ void im2col_kernel(const float* image, float* data_col,
//                             int height_in, int width_in,
//                             int height_kernel, int width_kernel,
//                             int height_out, int width_out,
//                             int channel_in, int stride, int pad_h, int pad_w) {
//     int i_out = blockIdx.y * blockDim.y + threadIdx.y;
//     int j_out = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i_out < height_out && j_out < width_out) {
//         for (int c = 0; c < channel_in; ++c) {
//             for (int i_kernel = 0; i_kernel < height_kernel; ++i_kernel) {
//                 for (int j_kernel = 0; j_kernel < width_kernel; ++j_kernel) {
//                     int h = i_out * stride - pad_h + i_kernel;
//                     int w = j_out * stride - pad_w + j_kernel;

//                     int image_index = (c * height_in + h) * width_in + w;
//                     int data_col_index = ((i_out * width_out + j_out) * channel_in + c) * height_kernel * width_kernel +
//                                          i_kernel * width_kernel + j_kernel;

//                     // Perform vectorized load and store if possible
//                     data_col[data_col_index] = (h >= 0 && h < height_in && w >= 0 && w < width_in) ?
//                                                 image[image_index] : 0.0;
//                 }
//             }
//         }
//     }
// }

// // Wrapper function for GPU im2col
// void im2col_gpu(const float* image, float* data_col,
//                 int height_in, int width_in,
//                 int height_kernel, int width_kernel,
//                 int height_out, int width_out,
//                 int channel_in, int stride, int pad_h, int pad_w) {
//     int block_size_x = 16;  // You can adjust this based on your device capabilities
//     int block_size_y = 16;  // You can adjust this based on your device capabilities

//     int grid_size_x = (width_out + block_size_x - 1) / block_size_x;
//     int grid_size_y = (height_out * channel_in * height_kernel + block_size_y - 1) / block_size_y;

//     dim3 block_dim(block_size_x, block_size_y);
//     dim3 grid_dim(grid_size_x, grid_size_y);

//     im2col_kernel<<<grid_dim, block_dim>>>(image, data_col,
//                                             height_in, width_in,
//                                             height_kernel, width_kernel,
//                                             height_out, width_out,
//                                             channel_in, stride, pad_h, pad_w);

//     cudaDeviceSynchronize();
// }
