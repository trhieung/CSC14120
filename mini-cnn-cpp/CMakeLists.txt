cmake_minimum_required (VERSION 3.0)
project (myproject)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Enable CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_executable(demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(demo MiniDNNLib)

# Link with CUDA
target_link_libraries(demo ${CUDA_LIBRARIES})

# Include CUDA headers
target_include_directories(demo PRIVATE ${CUDA_INCLUDE_DIRS})

# Add the CUDA source file to the project
set(CUDA_SOURCE_FILE "demo.cu")
set_source_files_properties(${CUDA_SOURCE_FILE} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
list(APPEND DIR_SRCS ${CUDA_SOURCE_FILE})
