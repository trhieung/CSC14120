#include <iostream>
#include "src/optimizer/sgd.h"

__global__ void myKernel(void) {
	printf("hello from kernel\n");
}

int main(void) {
	myKernel <<<2, 2>>>();
	cudaDeviceSynchronize()
	printf("Hello CUDA!\n");
	return 0;
}