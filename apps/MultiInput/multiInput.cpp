#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "trt_utils/common.h"

class SampleMultiInput {
  public:
    void build();
    void infer();

  private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtrTRT<nvinfer1::IExecutionContext> mContext{nullptr};
};

void SampleMultiInput::build() {
    // ----------------------------
    // Create builder and network
    // ----------------------------
    auto builder = UniquePtrTRT<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        UniquePtrTRT<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser =
        UniquePtrTRT<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(
        "./multi_input.onnx",
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = UniquePtrTRT<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config),
        InferDeleter());

    // ---------------
    // Create context
    // ---------------
    mContext = UniquePtrTRT<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
}

void SampleMultiInput::infer() {
    // -------------------
    // Check Tensornames
    // -------------------
    std::cout << std::endl << std::endl << "Check TensorNames" << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
        std::string tensorType = mEngine->bindingIsInput(i) ? "input: " : "output: ";
        std::cout << tensorType << mEngine->getBindingName(i) << std::endl;
    }
    std::cout << std::endl << std::endl;

    // -----------------------
    // Prepare Device Buffers
    // -----------------------
    int byteSize = 9 * 4;
    void *deviceBuffer_x1;
    void *deviceBuffer_x2;
    void *deviceBuffer_y;
    cudaMalloc(&deviceBuffer_x1, byteSize);
    cudaMalloc(&deviceBuffer_x2, byteSize);
    cudaMalloc(&deviceBuffer_y, byteSize);

    // ---------------------
    // Copy (Host -> Device)
    // ---------------------
    std::vector<float> x1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> x2{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    cudaMemcpy(deviceBuffer_x1, x1.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBuffer_x2, x2.data(), byteSize, cudaMemcpyHostToDevice);

    // --------
    // Execute
    // --------
    std::vector<void *> buffers{deviceBuffer_x1, deviceBuffer_x2, deviceBuffer_y};
    mContext->executeV2(buffers.data());

    // ----------------------
    // Copy (Device -> Host)
    // ----------------------
    std::vector<float> hostOutBuffer(9);
    cudaMemcpy(hostOutBuffer.data(), deviceBuffer_y, byteSize, cudaMemcpyDeviceToHost);

    // -------------
    // Print Result
    // -------------
    std::cout << "Input1" << std::endl;
    for (const auto &elem : x1) {
        std::cout << elem << std::endl;
    }
    std::cout << "Input2" << std::endl;
    for (const auto &elem : x2) {
        std::cout << elem << std::endl;
    }
    std::cout << "Result = Input1 + Input2" << std::endl;
    for (const auto &elem : hostOutBuffer) {
        std::cout << elem << std::endl;
    }

    // --------
    // Release
    // --------
    cudaFree(deviceBuffer_x1);
    cudaFree(deviceBuffer_x2);
    cudaFree(deviceBuffer_y);
}

int main() {
    SampleMultiInput sample;
    sample.build();
    sample.infer();
}
