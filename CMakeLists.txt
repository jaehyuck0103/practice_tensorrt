cmake_minimum_required(VERSION 3.9)

project(MOT_CPP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  # -Werror

find_package(OpenCV REQUIRED PATHS $ENV{HOME}/Util/opencv/build)

include_directories(/usr/local/cuda/include)
include_directories($ENV{HOME}/Util/TensorRT-7.1.3.4/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(third_party/eigen-3.3.8)
include_directories(third_party/nlohmann_json-3.9.1/include)

link_directories(/usr/local/cuda/lib64)
link_directories($ENV{HOME}/Util/TensorRT-7.1.3.4/lib)

add_executable(main1 apps/OnnxMNIST/sampleOnnx.cpp)
target_link_libraries(main1 cudart nvinfer nvonnxparser) 

add_executable(main2 apps/taillight/taillight.cpp)
target_link_libraries(main2 cudart nvinfer nvonnxparser ${OpenCV_LIBS} stdc++fs) 

add_executable(main3 apps/MultiInput/multiInput.cpp)
target_link_libraries(main3 cudart nvinfer nvonnxparser) 

add_executable(main4 apps/taillight2/main.cpp apps/taillight2/instance.cpp apps/taillight2/common.cpp)
target_link_libraries(main4 cudart nvinfer nvonnxparser ${OpenCV_LIBS}) 

add_executable(exec_BuildOnly apps/BuildOnly/main.cpp)
target_link_libraries(exec_BuildOnly cudart nvinfer nvonnxparser stdc++fs) 

add_executable(exec_benchmark apps/BuildOnly/benchmark.cpp)
target_link_libraries(exec_benchmark cudart nvinfer nvonnxparser stdc++fs) 

set(CMAKE_VERBOSE_MAKEFILE true)

add_custom_target(run1
    COMMAND main1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/OnnxMNIST  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run2
    COMMAND main2
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/taillight  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run3
    COMMAND main3
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/MultiInput  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run4
    COMMAND main4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/taillight2  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run4_imwrite
    COMMAND main4 --imwrite
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/taillight2  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run_BuildOnly
    COMMAND exec_BuildOnly
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/BuildOnly  # ${CMAKE_PROJECT_DIR}
    )

add_custom_target(run_benchmark
    COMMAND exec_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/BuildOnly  # ${CMAKE_PROJECT_DIR}
    )
