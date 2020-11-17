#pragma once
#include "BaseInferAgent.hpp"

#include "taillight/common.hpp"
#include <opencv2/opencv.hpp>

class RegressInferAgent : public BaseInferAgent {

  public:
    RegressInferAgent(const InferenceParams &params);
    std::vector<std::array<float, 4>> infer(const std::vector<cv::Mat> &croppedImgs);

  private:
};

inline RegressInferAgent::RegressInferAgent(const InferenceParams &params)
    : BaseInferAgent(params) {
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

inline std::vector<std::array<float, 4>>
RegressInferAgent::infer(const std::vector<cv::Mat> &croppedImgs) {
    std::vector<std::array<float, 4>> result;
    if (croppedImgs.empty()) {
        return result;
    }

    std::vector<float> hostInBuffer;
    hostInBuffer.reserve(RegCfg::inNumEl);
    const int realB = std::min(static_cast<int>(croppedImgs.size()), RegCfg::inB);
    for (int i = 0; i < realB; ++i) {
        if (!croppedImgs[i].isContinuous()) {
            std::cout << "Image is not continuous" << std::endl;
            exit(1);
        }
        if (croppedImgs[i].type() != CV_32FC3) {
            std::cout << "Invalid cv::Mat type" << std::endl;
            exit(1);
        }
        if (int(croppedImgs[i].total()) != (RegCfg::inH * RegCfg::inW)) {
            std::cout << "Invalid Input Feature Size" << std::endl;
            exit(1);
        }
        hostInBuffer.insert(hostInBuffer.end(), (float *)(croppedImgs[i].datastart),
                            (float *)(croppedImgs[i].dataend));
    }
    hostInBuffer.resize(RegCfg::inNumEl, 0.0f);

    // memcpy 버전
    /*
    const int eachNumel = RegCfg::inC * RegCfg::inH * RegCfg::inW;
    for (int i = 0; i < realB; ++i) {
        std::memcpy(&(hostInBuffer[i * eachNumel]), croppedImgs[i].data,
                    eachNumel * sizeof(float));
    }
    */

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
