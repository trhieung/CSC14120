// // #include <iostream>
// // #include "src/optimizer/sgd.h"
// // #include "src/kernel/HW0.h"
// // #include "src/kernel/HW0.cu"

// // __global__ void myKernel(void) {
// // 	printf("hello from kernel\n");
// // }

// int main(void) {
// 	myKernel <<<2, 2>>>();
// 	cudaDeviceSynchronize();
// 	printf("Hello CUDA!\n");
// 	printDeviceInfo();

// // 	//
// // 	MNIST dataset("CSC14120/mini-cnn-cpp/data/fashion-mnist/");
// //   dataset.read();
// //   int n_train = dataset.train_data.cols();
// //   int dim_in = dataset.train_data.rows();
// //   std::cout << "fashion-mnist train number: " << n_train << std::endl;
// //   std::cout << "fashion-mnist test number: " << dataset.test_labels.cols() << std::endl;
// //   //
// 	return 0;
// }

#include "src/kernel/Test.cuh"

int main(int argc, char *argv[])
{
	Wrapper::wrapper();
}