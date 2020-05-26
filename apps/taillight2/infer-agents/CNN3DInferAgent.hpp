#pragma once
#include "BaseInferAgent.hpp"
#include <list>

class CNN3DInferAgent : public BaseInferAgent {

  public:
    CNN3DInferAgent(const InferenceParams &params) : BaseInferAgent(params) {}
    std::vector<int> infer(const std::list<std::vector<float>> &encodedTails);

  private:
};

std::vector<int> CNN3DInferAgent::infer(const std::list<std::vector<float>> &encodedTailSeqs) {
    std::vector<int> result;
    if (encodedTailSeqs.empty()) {
        return result;
    }
    // ------------
    // Check Dims
    // ------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const int maxB = mEngine->getBindingDimensions(inputTensorIdx).d[0];     // 8
    const int inSeqLen = mEngine->getBindingDimensions(inputTensorIdx).d[1]; // 16
    const int inC = mEngine->getBindingDimensions(inputTensorIdx).d[2];      // 64
    const int inH = mEngine->getBindingDimensions(inputTensorIdx).d[3];      // 28
    const int inW = mEngine->getBindingDimensions(inputTensorIdx).d[4];      // 28

    const int outputTensorIdx = mEngine->getBindingIndex(mParams.outputTensorName.c_str());
    const int outMaxB = mEngine->getBindingDimensions(outputTensorIdx).d[0]; // 8
    const int outC = mEngine->getBindingDimensions(outputTensorIdx).d[1];    // 8

    if (maxB != 8 || inSeqLen != 16 || inC != 64 || inH != 28 || inW != 28) {
        std::cout << "Improper input tensor size1" << std::endl;
        exit(1);
    }
    if (outMaxB != 8 || outC != 1) {
        std::cout << "Improper input tensor size2" << std::endl;
        exit(1);
    }

    const int realB = std::min(static_cast<int>(encodedTailSeqs.size()), maxB);

    // -------------------
    // Prepare Input Data
    // -------------------
    std::vector<float> hostInBuffer;
    hostInBuffer.reserve(maxB * inSeqLen * inC * inH * inW);
    for (const auto &elem : encodedTailSeqs) {
        if (int(elem.size()) != inSeqLen * inC * inH * inW) {
            std::cout << "Invalid Input Feature Size" << std::endl;
            exit(1);
        }

        hostInBuffer.insert(hostInBuffer.end(), elem.begin(), elem.end());
    }
    hostInBuffer.resize(maxB * inSeqLen * inC * inH * inW, 0.0f);

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
    std::vector<int> hostOutBuffer(outMaxB * outC);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    std::cout << "result" << std::endl;
    for (int i = 0; i < realB; ++i) {
        std::cout << hostOutBuffer[i];
        /*
        for (int j = 0; j < outC; ++j) {
            std::cout << hostOutBuffer[i * 8 + j] << " ";
        }
        */
        std::cout << std::endl;
        result.emplace_back(hostOutBuffer[i]);
    }

    return result;
}
