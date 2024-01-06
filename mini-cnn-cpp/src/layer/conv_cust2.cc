// conv_cust2.cc

#include "conv_cust2.h"
#include <math.h>
#include "../kernel/HW2.cuh"
#include <iostream>

void ConvCust2::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void ConvCust2::im2col(const Vector& image, Matrix& data_col) {
  // TODO: Implement the im2col function for CPU
}

void ConvCust2::forward(const Matrix& bottom) {
  // TODO: Implement the forward function for CPU
}

void ConvCust2::backward(const Matrix& bottom, const Matrix& grad_top) {
  // TODO: Implement the backward function for CPU
}

void ConvCust2::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> ConvCust2::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void ConvCust2::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> ConvCust2::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
