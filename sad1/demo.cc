#include <iostream>
#include "src/mnist.h"


__global__ void myKernel(vector<int> x) {
	printf("hello from kernel, %d\n", x[0]);
}

int main(void) {
	vector<int> x {1, 2, 3, 4, 5}
	myKernel <<<2, 2>>>(x);
	cudaDeviceSynchronize();
	printf("Hello CUDA!\n");
	// printDeviceInfo();

// 	//
// 	MNIST dataset("CSC14120/mini-cnn-cpp/data/fashion-mnist/");
//   dataset.read();
//   int n_train = dataset.train_data.cols();
//   int dim_in = dataset.train_data.rows();
//   std::cout << "fashion-mnist train number: " << n_train << std::endl;
//   std::cout << "fashion-mnist test number: " << dataset.test_labels.cols() << std::endl;
//   //
	return 0;
}
