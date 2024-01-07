/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

// Include CUDA headers
#include "./src/kernel/Check.cuh"
#include "./src/kernel/GpuTimer.cuh"
#include "./src/kernel/HW0.cuh"
#include "./src/kernel/HW2.cuh"

int main() {
  // data
  // MNIST dataset("CSC14120/mini-cnn-cpp/data/mnist/");
  MNIST dataset("CSC14120/mini-cnn-cpp-gpu/data/fashion-mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "fashion-mnist train number: " << n_train << std::endl;
  std::cout << "fashion-mnist test number: " << dataset.test_labels.cols() << std::endl;
  
  // cnn
  Network dnn;
  Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 2, 2);
  Layer* pool2 = new MaxPooling(6, 28, 28, 2, 2, 2);
  Layer* conv3 = new Conv(6, 14, 14, 16, 5, 5, 1, 0, 0);
  Layer* pool4 = new MaxPooling(16, 10, 10, 2, 2, 2);
  Layer* fc5 = new FullyConnected(pool4->output_dim(), 120);
  Layer* fc6 = new FullyConnected(120, 84);
  Layer* fc7 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* relu5 = new ReLU;
  Layer* relu6 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool2);
  dnn.add_layer(relu2);
  dnn.add_layer(conv3);
  dnn.add_layer(relu3);
  dnn.add_layer(pool4);
  dnn.add_layer(relu4);
  dnn.add_layer(fc5);
  dnn.add_layer(relu5);
  dnn.add_layer(fc6);
  dnn.add_layer(relu6);
  dnn.add_layer(fc7);
  dnn.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  // SGD opt(0.001, 5e-4, 0.9, true);
  SGD opt(0.005, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  std::cout << "gpu batch size: " << batch_size << std::endl;

  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        dnn.check_gradient(x_batch, target_batch, 10);
      }
      dnn.forward(x_batch);
      dnn.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0) {
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
        << std::endl;
      }
      // optimize
      dnn.update(opt);
    }
    // test
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}

