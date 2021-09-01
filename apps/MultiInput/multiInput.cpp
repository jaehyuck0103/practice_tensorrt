#include "TensorRT-OSS/samples/common/common.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class SampleMultiInput {
  public:
    bool build();
    void infer();

  private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
};

bool SampleMultiInput::build() {
    // ----------------------------
    // Create builder and network
    // ----------------------------
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

    if (!builder) {
        return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // ------------------------
    // Create builder config
    // ------------------------
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }
    auto parsed = parser->parseFromFile(
        "./multi_input.onnx",
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        return false;
    }

    // -------------
    // Build engine
    // -------------
    std::unique_ptr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine) {
        return false;
    }

    // ---------------
    // Create context
    // ---------------
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    return true;
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
