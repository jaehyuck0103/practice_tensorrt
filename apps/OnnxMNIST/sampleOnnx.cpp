#include "TensorRT-OSS/samples/common/buffers.h"
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

struct SampleParams {
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{false}; //!< Allow running the network in FP16 mode.
    std::string inputTensorName;
    std::string outputTensorName;
    std::string onnxFilePath;
    std::string inputFilePath;
};

class SampleOnnxMNIST {

  public:
    SampleOnnxMNIST(const SampleParams &params) : mParams(params) {}

    bool build();
    void infer();

  private:
    SampleParams mParams;

    std::unique_ptr<samplesCommon::BufferManager> mBufManager{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
};

bool SampleOnnxMNIST::build() {
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
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    /*
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    */

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto parsed = parser->parseFromFile(
        mParams.onnxFilePath.c_str(),
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

    // -----------------------
    // Create buffer manager
    // -----------------------
    mBufManager = std::make_unique<samplesCommon::BufferManager>(mEngine);

    // ---------------
    // Create context
    // ---------------
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    return true;
}

void SampleOnnxMNIST::infer() {
    // -------------------
    // Prepare Input Data
    // -------------------
    int inputIndex = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const int inputH = mEngine->getBindingDimensions(inputIndex).d[2];
    const int inputW = mEngine->getBindingDimensions(inputIndex).d[3];

    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(mParams.inputFilePath, fileData.data(), inputH, inputW);

    float *hostInBuffer =
        static_cast<float *>(mBufManager->getHostBuffer(mParams.inputTensorName));
    for (int i = 0; i < inputH * inputW; i++) {
        hostInBuffer[i] = 1.0 - fileData[i] / 255.0;
    }

    // ----------------------
    // Copy (Host -> Device)
    // ----------------------
    mBufManager->copyInputToDevice();

    // --------
    // Execute
    // --------
    mContext->executeV2(mBufManager->getDeviceBindings().data());

    // ----------------------
    // Copy (Device -> Host)
    // ----------------------
    mBufManager->copyOutputToHost();

    // -------------
    // Print Result
    // -------------
    mBufManager->dumpBuffer(std::cout, mParams.outputTensorName);
}

int main() {
    SampleParams params;
    params.onnxFilePath = "./data/mnist.onnx";
    params.inputFilePath = "./data/8.pgm";
    params.inputTensorName = "Input3";
    params.outputTensorName = "Plus214_Output_0";

    SampleOnnxMNIST sample(params);
    sample.build();
    sample.infer();
}
