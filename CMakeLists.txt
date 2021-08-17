cmake_minimum_required(VERSION 3.17)

project(my_trt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

find_package(CUDAToolkit)

set(TRT_ROOT_DIR "$ENV{HOME}/Utils/TensorRT-7.2.3.4")
set(TRT_INCLUDE_DIR "${TRT_ROOT_DIR}/include")
set(TRT_LIBRARY_DIR "${TRT_ROOT_DIR}/lib")

add_library(my_trt INTERFACE)

target_include_directories(my_trt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                            ${TRT_INCLUDE_DIR})
target_link_directories(my_trt INTERFACE ${TRT_LIBRARY_DIR})
target_link_libraries(my_trt INTERFACE CUDA::cudart nvinfer nvonnxparser)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(apps/OnnxMNIST)
  add_subdirectory(apps/BuildOnly)
  add_subdirectory(apps/MultiInput)
endif()
