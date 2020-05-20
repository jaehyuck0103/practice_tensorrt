#pragma once
#include "BaseInferAgent.hpp"

#include <opencv2/opencv.hpp>

class UNetInferAgent : public BaseInferAgent {

  public:
    UNetInferAgent(const InferenceParams &params) : BaseInferAgent(params) {}
    std::vector<std::vector<float>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
};

std::vector<std::vector<float>> UNetInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::vector<float>> result;
    if (croppedImgs.empty()) {
        return result;
    }
    // ------------
    // Check Dims
    // ------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const int maxB = mEngine->getBindingDimensions(inputTensorIdx).d[0];     // 8
    const int inSeqLen = mEngine->getBindingDimensions(inputTensorIdx).d[1]; // 1
    const int inC = mEngine->getBindingDimensions(inputTensorIdx).d[2];      // 3
    const int inH = mEngine->getBindingDimensions(inputTensorIdx).d[3];      // 112
    const int inW = mEngine->getBindingDimensions(inputTensorIdx).d[4];      // 112

    const int outputTensorIdx = mEngine->getBindingIndex(mParams.outputTensorName.c_str());
    const int outMaxB = mEngine->getBindingDimensions(outputTensorIdx).d[0];   // 8
    const int outSeqLen = mEngine->getBindingDimensions(outputTensorIdx).d[1]; // 1
    const int outC = mEngine->getBindingDimensions(outputTensorIdx).d[2];      // 64
    const int outH = mEngine->getBindingDimensions(outputTensorIdx).d[3];      // 28
    const int outW = mEngine->getBindingDimensions(outputTensorIdx).d[4];      // 28

    if (maxB != 8 || inSeqLen != 1 || inC != 3 || inH != 112 || inW != 112) {
        std::cout << "Improper input tensor size" << std::endl;
        exit(1);
    }
    if (outMaxB != 8 || outSeqLen != 1 || outC != 64 || outH != 28 || outW != 28) {
        std::cout << "Improper input tensor size" << std::endl;
        exit(1);
    }

    // -------------------
    // Prepare Input Data
    // -------------------
    std::vector<float> hostInBuffer(maxB * inC * inH * inW);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), maxB);
    int idx = 0;
    for (int b = 0; b < realB; ++b) {
        for (int c = 0; c < inC; ++c) {
            for (int h = 0; h < inH; ++h) {
                for (int w = 0; w < inW; ++w) {
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
    std::vector<float> hostOutBuffer(outMaxB * outSeqLen * outC * outH * outW);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    const int eachBatchSize = outSeqLen * outC * outH * outW;
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
