cmake_minimum_required (VERSION 3.0)
project (myproject)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
find_package(CUDA REQUIRED)
# Add the following line to include the CUDA include directory
include_directories(${CUDA_INCLUDE_DIRS})
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)

# Set CUDA architecture (adjust as needed)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_75)

# 指定生成目标 
add_executable(demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(demo MiniDNNLib)