# Collect source files
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(layer DIR_LIB_SRCS)
aux_source_directory(loss DIR_LIB_SRCS)
aux_source_directory(optimizer DIR_LIB_SRCS)
aux_source_directory(kernel DIR_LIB_SRCS)

# Generate library
add_library(MiniDNNLib ${DIR_LIB_SRCS})

# CUDA files
file(GLOB_RECURSE CUDA_FILES "kernel/*.cu")
cuda_add_library(MiniDNNLibCUDA ${CUDA_FILES})
target_link_libraries(MiniDNNLib MiniDNNLibCUDA)
