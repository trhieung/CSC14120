cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(myproject LANGUAGES CXX CUDA)

enable_language(CUDA)
add_custom_target(run
    COMMAND demo ${ARGV}
    DEPENDS demo
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

aux_source_directory(. DIR_SRCS)
add_subdirectory(src)
add_executable(demo ${DIR_SRCS})
target_link_libraries(demo MiniDNNLib)


# cmake_minimum_required(VERSION 3.0)
# project(myproject)
# include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen")

# # Enable CUDA
# enable_language(CUDA)

# aux_source_directory(. DIR_SRCS)
# add_subdirectory(src)
# add_executable(demo ${DIR_SRCS})
# target_link_libraries(demo MiniDNNLib)

# # Set CUDA compute capability
# set(CMAKE_CUDA_ARCHITECTURES 75)

# # Find CUDA Toolkit and set include directories
# find_package(CUDA REQUIRED)
# target_include_directories(demo PRIVATE ${CUDA_INCLUDE_DIRS})
