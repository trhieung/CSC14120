#include "./final.cuh"

// Kernel function to perform im2col on GPU
__global__ void im2col_kernel(float* image, float* data_col,
                               int height_in, int width_in,
                               int height_kernel, int width_kernel,
                               int height_out, int width_out,
                               int channel_in, int stride) {
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < channel_in * hw_out) {
        int c = idx / hw_out;
        int i = idx % hw_out;

        int step_h = i / width_out;
        int step_w = i % width_out;
        int start_idx = step_h * width_in * stride + step_w * stride;

        for (int j = 0; j < hw_kernel; j++) {
            int cur_col = start_idx % width_in + j % width_kernel - (width_kernel / 2);
            int cur_row = start_idx / width_in + j / width_kernel - (height_kernel / 2);

            if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
                data_col[idx * hw_kernel + j] = 0;
            } else {
                int pick_idx = cur_row * width_in + cur_col;
                data_col[idx * hw_kernel + j] = image[c * hw_in + pick_idx];
            }
        }
    }
}

// Wrapper function to call im2col_kernel from CPU
void im2col_gpu(float* image, float* data_col,
                int height_in, int width_in,
                int height_kernel, int width_kernel,
                int height_out, int width_out,
                int channel_in, int stride) {
    int hw_out = height_out * width_out;
    int num_threads = 256;
    int num_blocks = (channel_in * hw_out + num_threads - 1) / num_threads;
    im2col_kernel<<<num_blocks, num_threads>>>(image, data_col,
                                               height_in, width_in,
                                               height_kernel, width_kernel,
                                               height_out, width_out,
                                               channel_in, stride);
}

void check() {
    // CPU
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;

    // GPU
    float* d_image;
    float* d_data_col;
    cudaMalloc((void**)&d_image, sizeof(float) * height_in * width_in * channel_in);
    cudaMalloc((void**)&d_data_col, sizeof(float) * height_out * width_out * channel_in * height_kernel * width_kernel);
    cudaMemcpy(d_image, bottom.col(i).data(), sizeof(float) * height_in * width_in * channel_in, cudaMemcpyHostToDevice);

    im2col_gpu(d_image, d_data_col, height_in, width_in, height_kernel, width_kernel, height_out, width_out, channel_in, stride);

    cudaMemcpy(data_cols[i].data(), d_data_col, sizeof(float) * height_out * width_out * channel_in * height_kernel * width_kernel, cudaMemcpyDeviceToHost);

    // Check equality (You may need to implement a function to compare CPU and GPU results)
    // ...
    
    cudaFree(d_image);
    cudaFree(d_data_col);
}
