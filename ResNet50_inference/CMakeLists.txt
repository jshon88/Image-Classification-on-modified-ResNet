cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ResNet50_inference)

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS} include)

add_executable(inference main.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")