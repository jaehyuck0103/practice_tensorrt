cmake_minimum_required(VERSION 3.17)

project(my_trt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

# CUDA
find_package(CUDAToolkit)

# TensorRT
set(TRT_ROOT_DIR "$ENV{HOME}/Utils/TensorRT-8.0.1.6")

add_library(trt INTERFACE)
target_include_directories(trt INTERFACE ${TRT_ROOT_DIR}/include)
target_link_directories(trt INTERFACE ${TRT_ROOT_DIR}/lib)
target_link_libraries(trt INTERFACE CUDA::cudart nvinfer nvparsers nvonnxparser)

# TensorRT-OSS
set(TRT_OSS_SAMPLES_COMMON_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT-OSS/samples/common")

add_library(trt_oss ${TRT_OSS_SAMPLES_COMMON_DIR}/logger.cpp
                    ${TRT_OSS_SAMPLES_COMMON_DIR}/sampleEngines.cpp)
target_include_directories(trt_oss
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(trt_oss PUBLIC trt)

# Build Apps
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(apps/OnnxMNIST)
  add_subdirectory(apps/BuildOnly)
  add_subdirectory(apps/MultiInput)
endif()
