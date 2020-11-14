#pragma once
#include "../common.hpp"
#include "BaseInferAgent.hpp"
#include <list>

class CNN3DInferAgent : public BaseInferAgent {

  public:
    CNN3DInferAgent(const InferenceParams &params);
    std::vector<int> infer(const std::list<std::vector<float>> &encodedTailSeqs);

  private:
};

inline CNN3DInferAgent::CNN3DInferAgent(const InferenceParams &params) : BaseInferAgent(params) {
    // ------------
    // Check Dims
    // ------------
    const int inputTensorIdx = mEngine->getBindingIndex(mParams.inputTensorName.c_str());
    const nvinfer1::Dims inDims = mEngine->getBindingDimensions(inputTensorIdx);
    checkDims(inDims, CNN3DCfg::inDims);

    const int outputTensorIdx = mEngine->getBindingIndex(mParams.outputTensorName.c_str());
    const nvinfer1::Dims outDims = mEngine->getBindingDimensions(outputTensorIdx);
    checkDims(outDims, CNN3DCfg::outDims);
}

inline std::vector<int>
CNN3DInferAgent::infer(const std::list<std::vector<float>> &encodedTailSeqs) {
    std::vector<int> result;
    if (encodedTailSeqs.empty()) {
        return result;
    }

    const int realB = std::min(static_cast<int>(encodedTailSeqs.size()), CNN3DCfg::inB);

    // -------------------
    // Prepare Input Data
    // -------------------
    std::vector<float> hostInBuffer;
    hostInBuffer.reserve(CNN3DCfg::inNumEl);
    for (const auto &elem : encodedTailSeqs) {
        if (int(elem.size()) != (CNN3DCfg::inNumEl / CNN3DCfg::inB)) {
            std::cout << "Invalid Input Feature Size" << std::endl;
            exit(1);
        }

        hostInBuffer.insert(hostInBuffer.end(), elem.begin(), elem.end());
    }
    hostInBuffer.resize(CNN3DCfg::inNumEl, 0.0f);

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
    std::vector<int> hostOutBuffer(CNN3DCfg::outNumEl);
    mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

    std::cout << "result" << std::endl;
    for (int i = 0; i < realB; ++i) {
        std::cout << hostOutBuffer[i] << std::endl;
        result.emplace_back(hostOutBuffer[i]);
    }

    return result;
}
