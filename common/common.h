#pragma once

#include <iostream>
#include <memory>
#include <numeric>

#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
        // suppress info-level messages
        if (severity != nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

struct InferDeleter {
    template <typename T> void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T> using UniquePtrTRT = std::unique_ptr<T, InferDeleter>;

inline int volume(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

inline int getTypeSize(nvinfer1::DataType t) {
    switch (t) {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    std::cout << "Invalid DataType." << std::endl;
    exit(1);
}
