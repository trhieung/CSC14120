#include <iostream>
__global__ void myKernel(void) {
	std::cout << "hello from kernel" << std::endl;
}
int main(void) {
	myKernel <<<2, 2>>>();
	printf("Hello CUDA!\n");
	return 0;
}