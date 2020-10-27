#pragma once
#include "../common.hpp"
#include "BaseInferAgent.hpp"

#include <opencv2/opencv.hpp>

class UNetInferAgent : public BaseInferAgent {

  public:
    UNetInferAgent(const InferenceParams &params);
    std::vector<std::vector<float>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
};

UNetInferAgent::UNetInferAgent(const InferenceParams &params) : BaseInferAgent(params) {
    // ------------
    // Check Dims
    // ------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const nvinfer1::Dims inDims = mEngine->getBindingDimensions(inputTensorIdx);
    checkDims(inDims, UNetCfg::inDims);

    const int outputTensorIdx = mEngine->getBindingIndex(mParams.outputTensorName.c_str());
    const nvinfer1::Dims outDims = mEngine->getBindingDimensions(outputTensorIdx);
    checkDims(outDims, UNetCfg::outDims);
}

std::vector<std::vector<float>> UNetInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::vector<float>> result;
    if (croppedImgs.empty()) {
        return result;
    }
    // -------------------
    // Prepare Input Data
    // -------------------
    std::vector<float> hostInBuffer;
    hostInBuffer.reserve(UNetCfg::inNumEl);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), UNetCfg::inB);
    for (int i = 0; i < realB; ++i) {
        if (!croppedImgs[i].isContinuous()) {
            std::cout << "Image is not continuous" << std::endl;
            exit(1);
        }
        if (croppedImgs[i].type() != CV_32FC3) {
            std::cout << "Invalid cv::Mat type" << std::endl;
            exit(1);
        }
        if (int(croppedImgs[i].total()) != (UNetCfg::inH * UNetCfg::inW)) {
            std::cout << "Invalid Input Feature Size" << std::endl;
            exit(1);
        }
        hostInBuffer.insert(hostInBuffer.end(), (float *)(croppedImgs[i].datastart),
                            (float *)(croppedImgs[i].dataend));
    }
    hostInBuffer.resize(UNetCfg::inNumEl, 0.0f);

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
    std::vector<float> hostOutBuffer(UNetCfg::outNumEl);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    const int eachBatchSize = UNetCfg::outNumEl / UNetCfg::outB;
    for (int i = 0; i < realB; ++i) {
        const auto iterBegin = hostOutBuffer.begin() + (i * eachBatchSize);
        const auto iterEnd = iterBegin + eachBatchSize;
        if (iterEnd > hostOutBuffer.end()) {
            std::cout << "iterator out of range" << std::endl;
            exit(1);
        }
        result.emplace_back(iterBegin, iterEnd);
    }

    return result;
}
