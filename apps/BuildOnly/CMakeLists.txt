add_executable(execBuildOnly buildOnly.cpp)
target_link_libraries(execBuildOnly PRIVATE trt_oss)

add_executable(execBenchmark benchmark.cpp)
target_link_libraries(execBenchmark PRIVATE trt_oss)

add_custom_target(
  runBuildOnly
  COMMAND execBuildOnly
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(
  runBenchmark
  COMMAND execBenchmark
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
