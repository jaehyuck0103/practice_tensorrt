add_library(libTaillight STATIC src/instance.cpp src/TailRecogManager.cpp)
target_include_directories(libTaillight
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libTaillight PRIVATE cudart nvinfer nvonnxparser
                                           ${OpenCV_LIBS})
