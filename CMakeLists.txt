cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_C_COMPILER "gcc-11")
set(CMAKE_CXX_COMPILER "g++-11")
set(CMAKE_PREFIX_PATH "../pytorch-install-2")

project(memory_machine)

set(EXECUTABLE_OUTPUT_PATH "../bin")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(memory_machine source.cpp)
set_target_properties(memory_machine PROPERTIES OUTPUT_NAME "machine1")
target_link_libraries(memory_machine "${TORCH_LIBRARIES}")
set_property(TARGET memory_machine PROPERTY CXX_STANDARD 17)

