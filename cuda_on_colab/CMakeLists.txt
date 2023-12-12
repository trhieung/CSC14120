cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)
add_custom_target(run
    COMMAND hello ${ARGV}
    DEPENDS hello
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
add_executable(hello hello.cu)

# cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# project(hello LANGUAGES CXX CUDA)
# include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# enable_language(CUDA)
# add_custom_target(run
#     COMMAND hello ${ARGV}
#     DEPENDS hello
#     WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
# )

# aux_source_directory(. DIR_SRCS)
# add_subdirectory(src)
# add_executable(hello ${DIR_SRCS})
# target_link_libraries(hello MiniDNNLib)