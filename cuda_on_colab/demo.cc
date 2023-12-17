#include <iostream>
#include "src/kernel/cuda_file.cuh"  // Include the CUDA header file

int main() {
    std::cout << "Hello, world!" << std::endl;

    // Call the CUDA function
    runCudaFunction();

    return 0;
}