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