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
    std::vector<float> hostInBuffer(UNetCfg::inNumEl);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), UNetCfg::inB);
    int idx = 0;
    for (int b = 0; b < realB; ++b) {
        for (int c = 0; c < UNetCfg::inC; ++c) {
            for (int h = 0; h < UNetCfg::inH; ++h) {
                for (int w = 0; w < UNetCfg::inW; ++w) {
                    hostInBuffer[idx] = croppedImgs[b].at<cv::Vec3f>(h, w)[c];
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
