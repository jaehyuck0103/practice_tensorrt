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

#include "trt_utils/bufferManager.h"
#include "trt_utils/common.h"

namespace fs = std::experimental::filesystem;
namespace chrono = std::chrono;
typedef chrono::high_resolution_clock hrc;
typedef chrono::duration<double, std::milli> duration_ms;

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
        runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr),
        InferDeleter());

    // -----------------------
    // Create buffer manager
    // -----------------------
    std::unique_ptr<BufferManager> bufManager = std::make_unique<BufferManager>(engine);

    // ---------------
    // Create context
    // ---------------
    UniquePtrTRT<nvinfer1::IExecutionContext> context =
        UniquePtrTRT<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // Iterate Iteration
    for (int iter = 0; iter < 10; ++iter) {

        std::cout << std::endl;
        const hrc::time_point t1_total = hrc::now();

        // Upload dummy inputs (Host -> Device)
        for (int idx = 0; idx < engine->getNbBindings(); idx++) {
            if (engine->bindingIsInput(idx)) {

                const hrc::time_point t1_dummy = hrc::now();
                int vol = volume(engine->getBindingDimensions(idx));
                std::cout << "upload vol: " << vol << std::endl;
                std::vector<float> hostInBuffer(vol, 0.0);
                for (int bufIdx = 0; bufIdx < vol; ++bufIdx) {
                    if (bufIdx % 2 == 0) {
                        hostInBuffer[bufIdx] = 1.0;
                    } else {
                        hostInBuffer[bufIdx] = -1.0;
                    }
                }
                const hrc::time_point t2_dummy = hrc::now();
                const duration_ms duration_dummy = t2_dummy - t1_dummy;
                std::cout << "dummy_time (ms): " << duration_dummy.count() << std::endl;

                const hrc::time_point t1_upload = hrc::now();
                bufManager->memcpy(true, engine->getBindingName(idx), hostInBuffer.data());
                const hrc::time_point t2_upload = hrc::now();
                const duration_ms duration_upload = t2_upload - t1_upload;
                std::cout << "upload_time (ms): " << duration_upload.count() << std::endl;
            }
        }

        // Execute
        const hrc::time_point t1_exec = hrc::now();
        std::vector<void *> buffers = bufManager->getDeviceBindings();
        context->executeV2(buffers.data());
        const hrc::time_point t2_exec = hrc::now();
        const duration_ms duration_exec = t2_exec - t1_exec;
        std::cout << "exec_time (ms): " << duration_exec.count() << std::endl;

        // Show dummy outputs (Device -> Host)
        const hrc::time_point t1_download_all = hrc::now();
        for (int idx = 0; idx < engine->getNbBindings(); idx++) {
            if (!engine->bindingIsInput(idx)) {
                int vol = volume(engine->getBindingDimensions(idx));
                std::cout << "download vol: " << vol << std::endl;
                std::vector<float> hostOutBuffer(vol);

                const hrc::time_point t1_download = hrc::now();
                bufManager->memcpy(false, engine->getBindingName(idx), hostOutBuffer.data());
                const hrc::time_point t2_download = hrc::now();
                const duration_ms duration_download = t2_download - t1_download;
                std::cout << "download_time (ms): " << duration_download.count() << std::endl;

                int printLen = std::min(int(hostOutBuffer.size()), 10);
                for (int p = 0; p < printLen; ++p) {
                    std::cout << hostOutBuffer[p] << " ";
                }
                std::cout << std::endl;
            }
        }
        const hrc::time_point t2_download_all = hrc::now();
        const duration_ms duration_download_all = t2_download_all - t1_download_all;
        std::cout << "download_all_time (ms): " << duration_download_all.count() << std::endl;

        hrc::time_point t2_total = hrc::now();
        const duration_ms duration_total = t2_total - t1_total;
        std::cout << "total_time (ms): " << duration_total.count() << std::endl;
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
