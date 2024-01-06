// conv_cust2.h

#ifndef SRC_LAYER_CONV_CUST2_H_
#define SRC_LAYER_CONV_CUST2_H_

#include <vector>
#include "../layer.h"

class ConvCust2 : public Layer {
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

  Matrix weight;
  Vector bias;
  Matrix grad_weight;
  Vector grad_bias;

  std::vector<Matrix> data_cols;

  void init();

 public:
  ConvCust2(int channel_in, int height_in, int width_in, int channel_out,
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

  void im2col_gpu(const Vector& image, Matrix& data_col_gpu);
  void forward_gpu(const Matrix& bottom);
};

#endif  // SRC_LAYER_CONV_CUST2_H_
