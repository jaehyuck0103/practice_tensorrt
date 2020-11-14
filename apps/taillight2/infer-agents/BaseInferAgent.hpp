#pragma once
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "trt_utils/bufferManager.h"

struct InferenceParams {
    std::string inputTensorName = "Input";
    std::string outputTensorName = "Output";
    std::string trtFilePath;
};

inline void checkDims(const nvinfer1::Dims &dims, std::vector<int> targetDims) {

    if (dims.nbDims != static_cast<int>(targetDims.size())) {
        std::cout << "Improper nbDims" << std::endl;
        exit(1);
    }

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] != targetDims[i]) {
            std::cout << "Improper input tensor size" << std::endl;
            std::cout << dims.d[i] << " (expected " << targetDims[i] << ")" << std::endl;
            exit(1);
        }
    }
}

class BaseInferAgent {

  public:
    BaseInferAgent(const InferenceParams &params) : mParams(params) { loadEngine(); }

    void loadEngine() {
        std::ifstream engineFile(mParams.trtFilePath, std::ios::binary);
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
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), InferDeleter());

        // -----------------------
        // Create buffer manager
        // -----------------------
        mBufManager = std::make_unique<BufferManager>(mEngine);

        // ---------------
        // Create context
        // ---------------
        mContext = UniquePtrTRT<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    };

  protected:
    InferenceParams mParams;

    std::unique_ptr<BufferManager> mBufManager{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtrTRT<nvinfer1::IExecutionContext> mContext{nullptr};
};
