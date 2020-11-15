add_executable(execTaillight main.cpp instance.cpp TailRecogManager.cpp)
target_link_libraries(execTaillight cudart nvinfer nvonnxparser ${OpenCV_LIBS})

add_custom_target(
  runTaillight
  COMMAND execTaillight
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(
  runTaillight_imwrite
  COMMAND execTaillight --imwrite
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
