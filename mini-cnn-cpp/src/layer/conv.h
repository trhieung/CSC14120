#ifndef SRC_LAYER_CONV_H_
#define SRC_LAYER_CONV_H_

#include <vector>
#include "../layer.h"

class Conv: public Layer {
 private:
  const int dim_in;
  int dim_out;

  int channel_in;
  int height_in;
  int width_in;
  int channel_out;
  int height_kernel;
  int width_kernel;
  int stride;
  int pad_h;
  int pad_w;

  int height_out;
  int width_out;

  Matrix weight;  // weight param, size=channel_in*h_kernel*w_kernel*channel_out
  Vector bias;  // bias param, size = channel_out
  Matrix grad_weight;  // gradient w.r.t weight
  Vector grad_bias;  // gradient w.r.t bias

  std::vector<Matrix> data_cols;

  void init();

 public:
  Conv(int channel_in, int height_in, int width_in, int channel_out,
       int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
       int pad_h = 0) :
       dim_in(channel_in * height_in * width_in),
       channel_in(channel_in), height_in(height_in), width_in(width_in),
       channel_out(channel_out), height_kernel(height_kernel),
       width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
  { init(); }

  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void update(Optimizer& opt);
  void im2col(const Vector& image, Matrix& data_col);
  void col2im(const Matrix& data_col, Vector& image);
  int output_dim() { return dim_out; }
  std::vector<float> get_parameters() const;
  std::vector<float> get_derivatives() const;
  void set_parameters(const std::vector<float>& param);

  void im2col_gpu(const Vector& image, Matrix& data_col_gpu) {
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
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(channel_in, (hw_out + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch GPU kernel
    im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(image_gpu, data_col_gpu_ptr, height_in, width_in,
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
};

#endif  // SRC_LAYER_CONV_H_
