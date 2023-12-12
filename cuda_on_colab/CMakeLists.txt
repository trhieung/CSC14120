cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)
add_custom_target(run
    COMMAND hello ${ARGV}
    DEPENDS hello
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
add_executable(hello hello.cu)