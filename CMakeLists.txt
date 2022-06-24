cmake_minimum_required(VERSION 3.18)

project(my_trt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

include(FetchContent)

# CUDA
find_package(CUDAToolkit)

# TensorRT
set(TRT_ROOT_DIR "$ENV{HOME}/Utils/TensorRT-8.4.1.5")

add_library(trt INTERFACE)
target_include_directories(trt INTERFACE ${TRT_ROOT_DIR}/include)
target_link_directories(trt INTERFACE ${TRT_ROOT_DIR}/lib)
target_link_libraries(trt INTERFACE CUDA::cudart nvinfer nvparsers nvonnxparser)

# TensorRT-OSS
FetchContent_Declare(
  trt-oss
  GIT_REPOSITORY https://github.com/NVIDIA/TensorRT
  GIT_TAG 8.4.1
  SOURCE_SUBDIR avoid_build)
FetchContent_MakeAvailable(trt-oss)

add_library(
  trt_oss
  ${trt-oss_SOURCE_DIR}/samples/common/logger.cpp
  ${trt-oss_SOURCE_DIR}/samples/common/sampleEngines.cpp
  ${trt-oss_SOURCE_DIR}/samples/common/sampleUtils.cpp
  ${trt-oss_SOURCE_DIR}/samples/common/sampleOptions.cpp)
target_include_directories(trt_oss INTERFACE ${CMAKE_BINARY_DIR}/_deps)
target_link_libraries(trt_oss PUBLIC trt ${CMAKE_DL_LIBS})

# Build Apps
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(apps/OnnxMNIST)
  add_subdirectory(apps/BuildOnly)
  add_subdirectory(apps/MultiInput)
endif()
