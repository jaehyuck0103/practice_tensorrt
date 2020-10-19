#pragma once
#include "BaseInferAgent.hpp"

#include "../common.hpp"
#include <opencv2/opencv.hpp>

class RegressInferAgent : public BaseInferAgent {

  public:
    RegressInferAgent(const InferenceParams &params);
    std::vector<std::array<float, 4>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
};

RegressInferAgent::RegressInferAgent(const InferenceParams &params) : BaseInferAgent(params) {
    // ------------
    // Check Dims
    // ------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const nvinfer1::Dims inDims = mEngine->getBindingDimensions(inputTensorIdx);
    checkDims(inDims, RegCfg::inDims);

    const int outputTensorIdx = mEngine->getBindingIndex(mParams.outputTensorName.c_str());
    const nvinfer1::Dims outDims = mEngine->getBindingDimensions(outputTensorIdx);
    checkDims(outDims, RegCfg::outDims);
}

std::vector<std::array<float, 4>>
RegressInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::array<float, 4>> result;
    if (croppedImgs.empty()) {
        return result;
    }

    std::vector<float> hostInBuffer(RegCfg::inNumEl);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), RegCfg::inB);
    /* memcpy 쓸까말까.... (쓰려면 network 시작을 BCHW -> BHWC로 바꿔줘야함)
    const int numel = inC * inH * inW;
    for (int i = 0; i < realBatch; ++i) {
        std::memcpy(&(hostInBuffer[i * numel]), croppedImgs[i].data, numel * sizeof(float));
    }
    */
    // 일단 안전하게 복사
    int idx = 0;
    for (int b = 0; b < realB; ++b) {
        for (int c = 0; c < RegCfg::inC; ++c) {
            for (int h = 0; h < RegCfg::inH; ++h) {
                for (int w = 0; w < RegCfg::inW; ++w) {
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
    std::vector<float> hostOutBuffer(RegCfg::outNumEl);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    for (int i = 0; i < realB; ++i) {
        result.push_back(std::array<float, 4>{hostOutBuffer[4 * i + 0], hostOutBuffer[4 * i + 1],
                                              hostOutBuffer[4 * i + 2], hostOutBuffer[4 * i + 3]});
    }

    return result;
}
