cmake_minimum_required(VERSION 3.9)

project(MOT_CPP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  # -Werror

include_directories(/usr/local/cuda/include)
include_directories($ENV{HOME}/Util/TensorRT-7.0.0.11/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(/usr/local/cuda/lib64)
link_directories($ENV{HOME}/Util/TensorRT-7.0.0.11/lib)

add_executable(main1 apps/OnnxMNIST/sampleOnnx.cpp)
target_link_libraries(main1 cudart nvinfer nvonnxparser) 

add_executable(main2 apps/taillight/taillight.cpp)
target_link_libraries(main2 cudart nvinfer nvonnxparser) 

set(CMAKE_VERBOSE_MAKEFILE true)

add_custom_target(run1
    COMMAND main1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/OnnxMNIST  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run2
    COMMAND main2
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/taillight  # ${CMAKE_PROJECT_DIR}
    )
