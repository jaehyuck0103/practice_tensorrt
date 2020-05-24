#include <cassert>
#include <chrono>
#include <experimental/filesystem> // gcc 8부터 experimental 뗄 수 있다. CMAKE의 link도 나중에 같이 떼주도록 하자.
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

namespace fs = std::experimental::filesystem;
namespace chrono = std::chrono;

struct SampleParams {
    std::string inputTensorName;
    std::string outputTensorName;
    std::vector<std::string> trtFilePath;
    std::string inputFileDir;
};

class TaillightInferenceAgent {

  public:
    TaillightInferenceAgent(const SampleParams &params) : mParams(params) {}

    void loadEngine(std::string trtFilePath);
    std::vector<float> infer(int idx, std::vector<float> hostInBuffer);
    void inferDummy();
    void inferTest();

  private:
    SampleParams mParams;
    const std::vector<std::string> mStates{"OOO", "BOO", "OLO", "BLO", "OOR", "BOR", "OLR", "BLR"};

    std::vector<std::unique_ptr<BufferManager>> mBufManagers;
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> mEngines;
    std::vector<UniquePtrTRT<nvinfer1::IExecutionContext>> mContexts;
};

void TaillightInferenceAgent::loadEngine(std::string trtFilePath) {
    std::ifstream engineFile(trtFilePath, std::ios::binary);
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
    mEngines.push_back(std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), InferDeleter()));

    // -----------------------
    // Create buffer manager
    // -----------------------
    mBufManagers.push_back(std::make_unique<BufferManager>(mEngines.back()));

    // ---------------
    // Create context
    // ---------------
    mContexts.push_back(
        UniquePtrTRT<nvinfer1::IExecutionContext>(mEngines.back()->createExecutionContext()));
}

std::vector<float> TaillightInferenceAgent::infer(int idx, std::vector<float> hostInBuffer) {
    std::string inputTensorName = "Input";
    std::string outputTensorName = "Output";

    // Get Vols
    const int inputTensorIdx = mEngines[idx]->getBindingIndex(inputTensorName.c_str());
    auto inputDims = mEngines[idx]->getBindingDimensions(inputTensorIdx);
    int inputVol = volume(inputDims);

    const int outputTensorIdx = mEngines[idx]->getBindingIndex(outputTensorName.c_str());
    auto outputDims = mEngines[idx]->getBindingDimensions(outputTensorIdx);
    int outputVol = volume(outputDims);

    std::vector<float> hostOutBuffer(outputVol);

    if (int(hostInBuffer.size()) != inputVol) {
        std::cout << "Wrong Input Size" << std::endl;
        exit(1);
    }

    // ----------------------
    // Copy (Host -> Device)
    // ----------------------
    mBufManagers[idx]->memcpy(true, inputTensorName, hostInBuffer.data());

    // -------------
    // Execute Unet
    // -------------
    std::vector<void *> buffers = mBufManagers[idx]->getDeviceBindings();
    mContexts[idx]->executeV2(buffers.data());

    // --------------------------------
    // Copy (Device -> Host -> Device)
    // --------------------------------
    mBufManagers[idx]->memcpy(false, outputTensorName, hostOutBuffer.data());

    return hostOutBuffer;
}

void TaillightInferenceAgent::inferDummy() {
    std::vector<float> input1(8 * 3 * 112 * 112);
    std::fill(input1.begin(), input1.end(), 0.1); // dummy test

    std::vector<float> output1 = infer(0, input1);

    std::vector<float> input2;
    for (int i = 0; i < 16; ++i) {
        input2.insert(input2.end(), output1.begin(), output1.end());
    }
    std::vector<float> output2 = infer(1, input2);

    for (const auto &elem : output2) {
        std::cout << elem << std::endl;
    }
}

void TaillightInferenceAgent::inferTest() {
    std::vector<float> input2;

    for (const auto &entryFolder : fs::directory_iterator(mParams.inputFileDir)) {
        // Collect img paths of a sequence
        std::vector<std::string> imgPaths;
        for (const auto &entryFile : fs::directory_iterator(entryFolder.path())) {
            imgPaths.push_back(entryFile.path());
        }
        std::sort(imgPaths.begin(), imgPaths.end());

        for (const auto &elem : imgPaths) {
            std::cout << elem << std::endl;
        }

        std::vector<cv::Mat> imgSeq;
        std::vector<float> output1;

        // load images
        for (const auto &elem : imgPaths) {
            cv::Mat img = cv::imread(elem);
            img.convertTo(img, CV_32FC3, 1.0 / 255.0);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            cv::resize(img, img, cv::Size(112, 112));
            img = (img - 0.5) * 4.0; // normalize
            imgSeq.push_back(img);
        }

        std::vector<float> input1(8 * 3 * 112 * 112);

        // first unet
        int idx = 0;
        for (int t = 0; t < 8; ++t) {
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 112; ++h) {
                    for (int w = 0; w < 112; ++w) {
                        input1[idx] = imgSeq[t].at<cv::Vec3f>(h, w)[c];
                        idx += 1;
                    }
                }
            }
        }
        output1 = infer(0, input1);
        input2.insert(input2.end(), output1.begin(), output1.end());

        // second unet
        idx = 0;
        for (int t = 8; t < 16; ++t) {
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 112; ++h) {
                    for (int w = 0; w < 112; ++w) {
                        input1[idx] = imgSeq[t].at<cv::Vec3f>(h, w)[c];
                        idx += 1;
                    }
                }
            }
        }
        output1 = infer(0, input1);
        input2.insert(input2.end(), output1.begin(), output1.end());
    }

    std::vector<float> output2 = infer(1, input2);

    std::cout << std::endl;
    int idx = 0;
    for (const auto &entryFolder : fs::directory_iterator(mParams.inputFileDir)) {
        std::cout << entryFolder << std::endl;
        for (int i = 0; i < 8; ++i) {
            std::cout << output2[idx] << " ";
            idx += 1;
        }
        std::cout << std::endl << std::endl;
    }
}

int main() {
    const std::string homeDir = std::getenv("HOME");
    SampleParams params;
    params.trtFilePath.push_back(
        homeDir +
        "/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_unet.trt");
    params.trtFilePath.push_back(
        homeDir +
        "/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_3Dconv.trt");

    params.inputFileDir =
        homeDir + "/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/TestInputs";
    params.inputTensorName = "Input";
    params.outputTensorName = "Output";

    TaillightInferenceAgent agent(params);
    agent.loadEngine(params.trtFilePath[0]);
    agent.loadEngine(params.trtFilePath[1]);

    // ----------
    // Infer Seq
    // ----------
    // agent.inferDummy();
    agent.inferTest();
    return 0;
}
