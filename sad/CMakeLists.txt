cmake_minimum_required(VERSION 3.17)
project(test)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
enable_language("CUDA")
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_SEPARABLE_COMPILATION ON)

string(APPEND CMAKE_CUDA_FLAGS " -rdc=true --cudart shared")

add_library(CUDA_COMP SHARED header.cuh kernel.cu)
set_property(TARGET CUDA_COMP PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)

add_executable(main main.cpp)

target_link_libraries(main CUDA_COMP)