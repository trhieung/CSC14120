#include <iostream>
#include "src/mnist.h"



int main(void) {
	MNIST dataset("CSC14120/sad1/data/fashion-mnist/");
	int n_train = dataset.train_data_col;
	int dim_in = dataset.train_data_dim.first * dataset.train_data_dim.second;
	std::cout << "fashion-mnist train number: " << n_train << std::endl;
  	std::cout << "fashion-mnist test number: " << dataset.test_data_col << std::endl;

	std::cout << "train dim: " << dim_in << std::endl;
  
	std::cout << "sad1" << std::endl;
    return 0;
}
