#include "TensorRT-OSS/samples/common/buffers.h"
#include "TensorRT-OSS/samples/common/common.h"
#include "TensorRT-OSS/samples/common/sampleEngines.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

struct SampleParams {
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

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<samplesCommon::BufferManager> mBufManager{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
};

bool SampleOnnxMNIST::build() {

    {
        sample::ModelOptions modelOption;
        modelOption.baseModel.model = mParams.onnxFilePath;
        modelOption.baseModel.format = sample::ModelFormat::kONNX;

        sample::BuildOptions buildOption;
        buildOption.workspace = 10 * 1024;
        buildOption.tf32 = false;
        buildOption.fp16 = true;
        buildOption.int8 = false;

        sample::SystemOptions sysOption;

        // Get Engine
        sample::BuildEnvironment env;
        getEngineBuildEnv(modelOption, buildOption, sysOption, env, std::cout);

        mEngine = std::move(env.engine);

        // network released after parser! parser destructor depends on network.
        env.parser = {};
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
