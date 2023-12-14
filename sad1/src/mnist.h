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

  void read();
 public:
  float* train_data;
  float* train_labels;
  float* test_data;
  float* test_labels;

  int train_data_col = 0;
  std::pair<int, int> train_data_dim = {0, 0};
  int test_data_col = 0;
  std::pair<int, int> test_data_dim = {0, 0};

  void get_cols_dim(std::string filename, std::pair<int, int>& dim, int& n_img); //dim = <rows, cols>
  
  void read_mnist_data(std::string filename, float*& data, const std::pair<int, int>& dim, const int& n_img);
  void read_mnist_label(std::string filename, float*& labels, const std::pair<int, int>& dim, const int& n_img);

  explicit MNIST(std::string data_dir) : data_dir(data_dir) {
    
    get_cols_dim(data_dir + "train-images-idx3-ubyte", train_data_dim, train_data_col);
    get_cols_dim(data_dir + "t10k-images-idx3-ubyte", test_data_dim, test_data_col);
    
    train_data = new float[train_data_col * train_data_dim.first *train_data_dim.second];
    train_labels = new float[train_data_col * 1];
    test_data = new float[test_data_col * test_data_dim.first * test_data_dim.second];
    test_labels = new float[test_data_col * 1];

    read();
  }
};

#endif  // SRC_MNIST_H_
