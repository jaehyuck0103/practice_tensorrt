cmake_minimum_required(VERSION 3.9)

project(my_trt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # -Werror
set(CMAKE_VERBOSE_MAKEFILE true)

find_package(OpenCV REQUIRED PATHS $ENV{HOME}/Utils/opencv/build)

include_directories(/usr/local/cuda/include)
include_directories($ENV{HOME}/Utils/TensorRT-7.2.3.4/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(third_party/eigen-3.3.8)
include_directories(third_party/nlohmann_json-3.9.1/include)
include_directories(third_party/toml11-3.6.0)

link_directories(/usr/local/cuda/lib64)
link_directories($ENV{HOME}/Utils/TensorRT-7.2.3.4/lib)

add_subdirectory(./modules/taillight)

add_subdirectory(./apps/OnnxMNIST)
add_subdirectory(./apps/BuildOnly)
add_subdirectory(./apps/taillight)
add_subdirectory(./apps/MultiInput)
