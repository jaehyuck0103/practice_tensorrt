add_executable(execMultiInput multiInput.cpp)
target_link_libraries(execMultiInput PRIVATE trt_oss)

add_custom_target(
  runMultiInput
  COMMAND execMultiInput
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
