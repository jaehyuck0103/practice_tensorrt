add_executable(execBuildOnly main.cpp)
target_link_libraries(execBuildOnly cudart nvinfer nvonnxparser stdc++fs)

add_executable(execBenchmark benchmark.cpp)
target_link_libraries(execBenchmark cudart nvinfer nvonnxparser stdc++fs)

add_custom_target(
  runBuildOnly
  COMMAND execBuildOnly
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(
  runBenchmark
  COMMAND execBenchmark
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
