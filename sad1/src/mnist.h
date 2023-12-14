#ifndef SRC_MNIST_H_
#define SRC_MNIST_H_

#include <fstream>
#include <iostream>
#include <string>
#include <utility> 
// #include "./utils.h"

class MNIST {
 private:
  std::string data_dir;

 public:
  // float* train_data;
  // float* train_labels;
  // float* test_data;
  // float* test_labels;

  // int train_data_col = 0;
  // int train_data_dim = 0;
  // int test_data_col = 0;
  // int test_data_dim = 0;

  void get_cols_dim(std::string filename, std::pair<int, int>& dim, int& n_img); //dim = <rows, cols>
  
  void read_mnist_data(std::string filename, float*& data, const std::pair<int, int>& dim, const int& n_img);
  void read_mnist_label(std::string filename, float*& labels, const std::pair<int, int>& dim, const int& n_img);

  explicit MNIST(std::string data_dir) : data_dir(data_dir) {}
  void read();
};

#endif  // SRC_MNIST_H_
