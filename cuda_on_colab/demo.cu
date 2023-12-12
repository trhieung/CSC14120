#include <iostream>
#include "src/optimizer/sgd.h"
#include "src/kernel/HW0.h"
#include "src/kernel/HW0.cu"

__global__ void myKernel(void) {
	printf("hello from kernel\n");
}

int main(void) {
	myKernel <<<2, 2>>>();
	cudaDeviceSynchronize()
	printf("Hello CUDA!\n");
	printDeviceInfo();
	return 0;
}