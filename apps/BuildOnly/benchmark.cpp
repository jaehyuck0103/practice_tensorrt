#include <algorithm>
#include <chrono>
#include <experimental/filesystem> // gcc 8부터 experimental 뗄 수 있다. CMAKE의 link도 나중에 같이 떼주도록 하자.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "common/bufferManager.h"
#include "common/common.h"

namespace fs = std::experimental::filesystem;
namespace chrono = std::chrono;

void benchmark(const std::string &trtFilePath) {
    // ------------
    // Load Engine
    // ------------
    std::ifstream engineFile(trtFilePath, std::ios::binary);
    if (engineFile.fail()) {
        std::cout << "Error opening TRT file." << std::endl;
        exit(1);
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (engineFile.fail()) {
        std::cout << "Error reading TRT file." << std::endl;
        exit(1);
    }

    UniquePtrTRT<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
    // if (DLACore != -1) { runtime->setDLACore(DLACore); }
    std::shared_ptr<nvinfer1::ICudaEngine> engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), InferDeleter());

    // -----------------------
    // Create buffer manager
    // -----------------------
    std::unique_ptr<BufferManager> bufManager = std::make_unique<BufferManager>(engine);

    // ---------------
    // Create context
    // ---------------
    UniquePtrTRT<nvinfer1::IExecutionContext> context =
        UniquePtrTRT<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // ------------
    // Run dummy
    // ------------

    // Put dummy inputs (Host -> Device)
    for (int idx = 0; idx < engine->getNbBindings(); idx++) {
        if (engine->bindingIsInput(idx)) {
            int vol = volume(engine->getBindingDimensions(idx));
            std::vector<float> hostInBuffer(vol, 0.0);
            for (int bufIdx = 0; bufIdx < vol; ++bufIdx) {
                if (bufIdx % 2 == 0) {
                    hostInBuffer[bufIdx] = 1.0;
                } else {
                    hostInBuffer[bufIdx] = -1.0;
                }
            }
            bufManager->memcpy(true, engine->getBindingName(idx), hostInBuffer.data());
        }
    }

    // Iterate Iteration
    for (int iter = 0; iter < 100; ++iter) {
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        // Execute
        std::vector<void *> buffers = bufManager->getDeviceBindings();
        context->executeV2(buffers.data());
        // Show dummy outputs (Device -> Host)
        for (int idx = 0; idx < engine->getNbBindings(); idx++) {
            if (!engine->bindingIsInput(idx)) {
                int vol = volume(engine->getBindingDimensions(idx));
                std::vector<float> hostOutBuffer(vol);
                bufManager->memcpy(false, engine->getBindingName(idx), hostOutBuffer.data());

                int printLen = std::min(int(hostOutBuffer.size()), 10);
                for (int p = 0; p < printLen; ++p) {
                    std::cout << hostOutBuffer[p] << " ";
                }
                std::cout << std::endl;
            }
        }
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        std::cout << "processing_time (micro sec): " << duration << std::endl;
    }
}

int main() {
    std::cout << std::endl << std::endl << std::endl << std::endl;

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
        std::cout << std::endl << std::endl << std::endl << std::endl;
        std::cout << elem << std::endl << std::endl;
        benchmark(elem);
    }
    std::cout << std::endl << std::endl << std::endl << std::endl;
}
