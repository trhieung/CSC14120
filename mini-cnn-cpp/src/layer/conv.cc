#include "conv.h"
#include <math.h>
#include "../kernel/HW2.cuh"
#include "../kernel/final.cuh"
#include <iostream>

void Conv::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c ++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
        }
      }
    }
  }
}
#include <iostream>
#include <cmath>

// Check function to compare GPU and CPU im2col results
bool checkIm2Col(const Matrix& cpu_result, const float* gpu_result, int size) {
  int cnt = 0;
    for (int i = 0; i < size; ++i) {
        if (std::abs(cpu_result(i) - gpu_result[i]) > 1e-5) {
            // std::cout << "Mismatch at index " << i << ": CPU = " << cpu_result(i)
            //           << ", GPU = " << gpu_result[i] << std::endl;
            // return false;
            cnt++;
        }
    }
    // std::cout << "Results match!" << std::endl;
    std::cout << "hihi: " << cnt << std::endl;
    return true;
}
void Conv::forward(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    std::cout << "huhu" << std::endl;
    // std::cout << data_col.row(10) << std::endl << std::endl;
    Matrix T = data_col.transpose();
    float* p = T.data();
    int _v = height_kernel * width_kernel * channel_in;

    for(int i = 0; i < _v; i++){
      std::cout << p[10*_v + i]<< " ";
    }

    // Matrix A(4, 5);
    // A << 1, 2, 3, 4, 5,
    //     6, 7, 8, 9, 10,
    //     11, 12, 13, 14, 15,
    //     16, 17, 18, 19, 20;

    // // Print the matrix
    // std::cout << "Matrix A (4x5):\n" << A << std::endl << std::endl;
    // std::cout << A.row(2) << std::endl << std::endl;
    // for(int i = 0; i < 5; i++){
    //   std::cout << p[2*5 +i ]<< " ";
    // }
    std::cout << std::endl << std::endl;
    // test
    // GPU
    float* d_image;
    float* d_data_col;
    int size_image = height_in * width_in * channel_in;
    int size_data_col = height_out * width_out * height_kernel * width_kernel * channel_in;

    // Allocate GPU memory
    cudaMalloc((void**)&d_image, size_image * sizeof(float));
    cudaMalloc((void**)&d_data_col, size_data_col * sizeof(float));

    // Transfer data from CPU to GPU
    cudaMemcpy(d_image, bottom.col(i).data(), size_image * sizeof(float), cudaMemcpyHostToDevice);

    // Call GPU im2col function
    im2col_gpu(d_image, d_data_col,
              height_in, width_in,
              height_kernel, width_kernel,
              height_out, width_out,
              channel_in, stride, pad_h, pad_w);

    // Transfer data from GPU to CPU
    float* data_col_gpu = new float[size_data_col];
    cudaMemcpy(data_col_gpu, d_data_col, size_data_col * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < _v; i++){
      std::cout << data_col_gpu[10*_v + i]<< " ";
    }

    // Check results
    checkIm2Col(T, data_col_gpu, size_data_col);

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_data_col);
    delete[] data_col_gpu;

    std::cout << std::endl << std::endl;
    // conv by product
    Matrix result = data_col * weight;  // result: (hw_out, channel_out)

    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv::col2im(const Matrix& data_col, Vector& image) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // col2im
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c ++) {
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
        }
      }
    }
  }
}

void Conv::backward(const Matrix& bottom, const Matrix& grad_top) {
  int n_sample = bottom.cols();
  grad_weight.setZero();
  grad_bias.setZero();
  grad_bottom.resize(height_in * width_in * channel_in, n_sample);
  grad_bottom.setZero();
  for (int i = 0; i < n_sample; i ++) {
    // im2col of grad_top
    Matrix grad_top_i = grad_top.col(i);
    Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                              height_out * width_out, channel_out);
    // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
    grad_weight += data_cols[i].transpose() * grad_top_i_col;
    // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
    grad_bias += grad_top_i_col.colwise().sum().transpose();
    // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
    Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
    // col2im of grad_bottom
    Vector grad_bottom_i;
    col2im(grad_bottom_i_col, grad_bottom_i);
    grad_bottom.col(i) = grad_bottom_i;
  }
}

void Conv::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}

//---------------------------------------------------------------

// void Conv::forward(const Matrix& bottom) {

//   int n_sample = bottom.cols();
//   top.resize(height_out * width_out * channel_out, n_sample);
//   data_cols.resize(n_sample);

//   // paralel init
//   dim3 blockSize(32, 32);
//   Matrix data_col_t;
//   Matrix result_t;
//   Matrix weight_t;
//   Matrix result;
//   float* _data_col;
//   float* _weight;
//   float* _correct_result;
//   float* _result = new float[height_out * width_out*channel_out];

//   for (int i = 0; i < n_sample; i ++) {
//     // im2col
//     Matrix data_col;
//     im2col(bottom.col(i), data_col);
//     data_cols[i] = data_col;
//     // conv by product
//     data_col_t  = data_col.transpose();
//     weight_t = weight.transpose();
//     _data_col = data_col_t.data(); //(hw_out, hw_kernel * channel_in)
//     _weight = weight_t.data();     //(channel_in * height_kernel * width_kernel, channel_out)
//     _correct_result = result_t.data();
    
//     GpuTimer timer;
//     timer.Start();

//     matrix_multiplication(_data_col, _weight, _result, height_out * width_out, channel_in * height_kernel * width_kernel, channel_out, true,blockSize,2);

//     timer.Stop();
//     float time = timer.Elapsed();
//     printf("Processing time (%s): %f ms\n", "use device", time);

//     result_t = Eigen::Map<Matrix>(_result, channel_out, height_out * width_out);
//     result = result_t.transpose();
//     result.rowwise() += bias.transpose();
//     top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
//   }
//   delete[] _result;
// }