#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "trt_utils/bufferManager.h"

struct SampleParams {
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{false}; //!< Allow running the network in FP16 mode.
    std::string inputTensorName;
    std::string outputTensorName;
    std::string onnxFilePath;
    std::string inputFilePath;
};

void readPGMFile(const std::string &fileName, uint8_t *buffer, int inH, int inW) {
    std::ifstream infile(fileName, std::ifstream::binary);
    if (!infile.is_open()) {
        std::cout << "Attempting to read from a file that is not open." << std::endl;
        exit(1);
    }
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char *>(buffer), inH * inW);
}

class SampleOnnxMNIST {

  public:
    SampleOnnxMNIST(const SampleParams &params) : mParams(params) {}

    void build();
    void infer();

  private:
    SampleParams mParams;

    std::unique_ptr<BufferManager> mBufManager{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtrTRT<nvinfer1::IExecutionContext> mContext{nullptr};
};

void SampleOnnxMNIST::build() {
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
        mParams.onnxFilePath.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = UniquePtrTRT<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    /*
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1 << 30);
    if (mParams.fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }
    if (mParams.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    */

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config),
        InferDeleter());

    // -----------------------
    // Create buffer manager
    // -----------------------
    mBufManager = std::make_unique<BufferManager>(mEngine);

    // ---------------
    // Create context
    // ---------------
    mContext = UniquePtrTRT<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
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

    std::vector<float> hostInBuffer(inputH * inputW);
    for (int i = 0; i < inputH * inputW; ++i) {
        hostInBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    // ----------------------
    // Copy (Host -> Device)
    // ----------------------
    mBufManager->memcpy(true, mParams.inputTensorName, hostInBuffer.data());

    // --------
    // Execute
    // --------
    std::vector<void *> buffers = mBufManager->getDeviceBindings();
    mContext->executeV2(buffers.data());

    // ----------------------
    // Copy (Device -> Host)
    // ----------------------
    std::vector<float> hostOutBuffer(10);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    // -------------
    // Print Result
    // -------------
    std::cout << "Result" << std::endl;
    for (const auto &elem : hostOutBuffer) {
        std::cout << elem << std::endl;
    }
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
