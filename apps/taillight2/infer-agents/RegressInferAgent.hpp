#pragma once
#include "BaseInferAgent.hpp"

#include <opencv2/opencv.hpp>

class RegressInferAgent : public BaseInferAgent {

  public:
    RegressInferAgent(const InferenceParams &params) : BaseInferAgent(params) {}
    std::vector<std::array<float, 4>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
};

std::vector<std::array<float, 4>>
RegressInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::array<float, 4>> result;
    if (croppedImgs.empty()) {
        return result;
    }
    // -------------------
    // Prepare Input Data
    // -------------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const int maxB = mEngine->getBindingDimensions(inputTensorIdx).d[0]; // 8
    const int inC = mEngine->getBindingDimensions(inputTensorIdx).d[1];  // 3
    const int inH = mEngine->getBindingDimensions(inputTensorIdx).d[2];  // 224
    const int inW = mEngine->getBindingDimensions(inputTensorIdx).d[3];  // 224

    if (maxB != 8 || inC != 3 || inH != 224 || inW != 224) {
        std::cout << "Improper input tensor size" << std::endl;
        exit(1);
    }

    std::vector<float> hostInBuffer(maxB * inC * inH * inW);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), maxB);
    /* memcpy 쓸까말까.... (쓰려면 network 시작을 BCHW -> BHWC로 바꿔줘야함)
    const int numel = inC * inH * inW;
    for (int i = 0; i < realBatch; ++i) {
        std::memcpy(&(hostInBuffer[i * numel]), croppedImgs[i].data, numel * sizeof(float));
    }
    */
    // 일단 안전하게 복사
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
    std::vector<float> hostOutBuffer(8 * 4);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    for (int i = 0; i < realB; ++i) {
        result.push_back(std::array<float, 4>{hostOutBuffer[4 * i + 0], hostOutBuffer[4 * i + 1],
                                              hostOutBuffer[4 * i + 2], hostOutBuffer[4 * i + 3]});
    }

    return result;
}
