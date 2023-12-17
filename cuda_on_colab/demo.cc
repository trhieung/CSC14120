#include <iostream>

// Include the CUDA header file
#include "./src/kernel/cuda_file.cuh"  
#include "./src/kernel/HW0.cuh"

int main() {
    std::cout << "Hello, world!" << std::endl;

    // Call the CUDA function
    runCudaFunction();
    printDeviceInfo();
    return 0;
}