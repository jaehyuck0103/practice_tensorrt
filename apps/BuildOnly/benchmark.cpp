#include "TensorRT-OSS/samples/common/buffers.h"
#include "TensorRT-OSS/samples/common/common.h"
#include "TensorRT-OSS/samples/common/sampleEngines.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace chrono = std::chrono;
typedef chrono::high_resolution_clock hrc;
typedef chrono::duration<double, std::milli> duration_ms;

bool benchmark(const std::string &trtFilePath) {
    // ------------
    // Load Engine
    // ------------
    std::shared_ptr<nvinfer1::ICudaEngine> engine{sample::loadEngine(trtFilePath, -1, std::cout)};
    if (!engine) {
        return false;
    }

    // -----------------------
    // Create buffer manager
    // -----------------------
    auto bufManager = std::make_unique<samplesCommon::BufferManager>(engine);

    // ---------------
    // Create context
    // ---------------
    std::unique_ptr<nvinfer1::IExecutionContext> context =
        std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // Measure multiple times
    for (int iter = 0; iter < 10; ++iter) {

        std::cout << std::endl;
        const hrc::time_point t1_total = hrc::now();

        // Put dummy inputs
        for (int idx = 0; idx < engine->getNbBindings(); idx++) {
            if (engine->bindingIsInput(idx)) {
                int vol = samplesCommon::volume(engine->getBindingDimensions(idx));
                std::cout << "upload vol: " << vol << std::endl;
                float *hostInBuffer =
                    static_cast<float *>(bufManager->getHostBuffer(engine->getBindingName(idx)));
                for (int bufIdx = 0; bufIdx < vol; ++bufIdx) {
                    if (bufIdx % 2 == 0) {
                        hostInBuffer[bufIdx] = 1.0;
                    } else {
                        hostInBuffer[bufIdx] = -1.0;
                    }
                }
            }
        }

        // Upload
        const hrc::time_point t1_upload = hrc::now();
        bufManager->copyInputToDevice();
        const hrc::time_point t2_upload = hrc::now();
        const duration_ms duration_upload = t2_upload - t1_upload;
        std::cout << "upload_time (ms): " << duration_upload.count() << std::endl;

        // Execute
        const hrc::time_point t1_exec = hrc::now();
        context->executeV2(bufManager->getDeviceBindings().data());
        const hrc::time_point t2_exec = hrc::now();
        const duration_ms duration_exec = t2_exec - t1_exec;
        std::cout << "exec_time (ms): " << duration_exec.count() << std::endl;

        // Download
        const hrc::time_point t1_download = hrc::now();
        bufManager->copyOutputToHost();
        const hrc::time_point t2_download = hrc::now();
        const duration_ms duration_download = t2_download - t1_download;
        std::cout << "download_time (ms): " << duration_download.count() << std::endl;

        // Show dummy outputs
        for (int idx = 0; idx < engine->getNbBindings(); idx++) {
            if (!engine->bindingIsInput(idx)) {
                const std::string tensorName = engine->getBindingName(idx);
                bufManager->print<float>(
                    std::cout,
                    bufManager->getHostBuffer(tensorName),
                    std::min(sizeof(float) * 10, bufManager->size(tensorName)),
                    10);
                std::cout << "\n";
            }
        }

        hrc::time_point t2_total = hrc::now();
        const duration_ms duration_total = t2_total - t1_total;
        std::cout << "total_time (ms): " << duration_total.count() << "\n\n";
    }

    return true;
}

int main() {
    std::cout << "\n\n\n\n\n";

    // Get Onnx lists
    const std::string homeDir = std::getenv("HOME");
    std::ifstream f("./InputOnnxList.txt");
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(f, line)) {
        if (line.rfind("~", 0) == 0) {
            line = homeDir + line.substr(1);
        }
        line = fs::path{line}.replace_extension(".trt");

        lines.push_back(line);
    }

    // Build each onnx
    for (const auto &elem : lines) {
        std::cout << "\n\n\n\n\n";
        std::cout << elem << "\n\n";
        benchmark(elem);
    }
    std::cout << "\n\n\n\n\n";
}
