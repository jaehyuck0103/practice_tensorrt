add_executable(execOnnxMNIST sampleOnnx.cpp)
target_link_libraries(execOnnxMNIST PRIVATE trt_oss)

add_custom_target(
  runOnnxMNIST
  COMMAND execOnnxMNIST
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
