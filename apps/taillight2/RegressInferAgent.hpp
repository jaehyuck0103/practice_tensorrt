#pragma once
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "common/bufferManager.h"

struct SampleParams {
    std::string inputTensorName;
    std::string outputTensorName;
    std::string trtFilePath;
};

class RegressInferAgent {

  public:
    RegressInferAgent(const SampleParams &params) : mParams(params) { loadEngine(); }

    void loadEngine();
    std::vector<std::array<float, 4>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
    SampleParams mParams;

    std::unique_ptr<BufferManager> mBufManager{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtrTRT<nvinfer1::IExecutionContext> mContext{nullptr};
};

void RegressInferAgent::loadEngine() {
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
}

std::vector<std::array<float, 4>>
RegressInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::array<float, 4>> result;
    if (croppedImgs.empty()) {
        return result;
    }
    // -------------------
    // Prepare Input Data
    // -------------------
    int inputIndex = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const int batchSize = mEngine->getBindingDimensions(inputIndex).d[0]; // 8
    const int inputH = 224; // mEngine->getBindingDimensions(inputIndex).d[2];
    const int inputW = 224; // mEngine->getBindingDimensions(inputIndex).d[3];

    std::vector<float> hostInBuffer(batchSize * 3 * inputH * inputW);
    const int realBatch = std::min(static_cast<int>(croppedImgs.size()), batchSize);
    /* memcpy 쓸까말까.... (쓰려면 network 시작을 BCHW -> BHWC로 바꿔줘야함)
    const int numel = inputH * inputW * 3;
    for (int i = 0; i < realBatch; ++i) {
        std::memcpy(&(hostInBuffer[i * numel]), croppedImgs[i].data, numel * sizeof(float));
    }
    */
    // 일단 안전하게 복사
    int idx = 0;
    for (int i = 0; i < realBatch; ++i) {
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 224; ++h) {
                for (int w = 0; w < 224; ++w) {
                    hostInBuffer[idx] = croppedImgs[i].at<cv::Vec3f>(h, w)[c];
                    idx += 1;
                }
            }
        }
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
    std::vector<float> hostOutBuffer(8 * 4);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    for (int i = 0; i < realBatch; ++i) {
        result.push_back(std::array<float, 4>{hostOutBuffer[4 * i + 0], hostOutBuffer[4 * i + 1],
                                              hostOutBuffer[4 * i + 2], hostOutBuffer[4 * i + 3]});
    }

    return result;
}
