# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)
aux_source_directory(./kernel DIR_LIB_SRCS)
# 生成链接库
add_library (MiniDNNLib ${DIR_LIB_SRCS})


# aux_source_directory(./kernel DIR_LIB_SRCS)

# # CUDA sources
# file(GLOB CUDA_SOURCES "kernel/*.cu")
# list(APPEND DIR_LIB_SRCS ${CUDA_SOURCES})

# # Set CUDA compute capability
# set(CMAKE_CUDA_ARCHITECTURES 75)

# add_library(MiniDNNLib ${DIR_LIB_SRCS})

# # Include CUDA toolkit headers
# target_include_directories(MiniDNNLib PRIVATE ${CUDA_INCLUDE_DIRS})
