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
    std::vector<std::string> onnxFilePath;
    std::string inputFileDir;
};

class TaillightInferenceAgent {

  public:
    TaillightInferenceAgent(const SampleParams &params) : mParams(params) {}

    void build(std::string onnxFilePath);
    std::vector<std::string> infer(std::vector<std::string> imgPaths);
    void benchmark_speed();

  private:
    SampleParams mParams;
    const std::vector<std::string> mStates{"OOO", "BOO", "OLO", "BLO", "OOR", "BOR", "OLR", "BLR"};

    std::vector<std::unique_ptr<BufferManager>> mBufManagers;
    // std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::vector<UniquePtrTRT<nvinfer1::IExecutionContext>> mContexts;
};

void TaillightInferenceAgent::build(std::string onnxFilePath) {
    // ----------------------------
    // Create builder and network
    // ----------------------------
    auto builder = UniquePtrTRT<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        UniquePtrTRT<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser =
        UniquePtrTRT<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(onnxFilePath.c_str(),
                          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = UniquePtrTRT<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30);

    std::shared_ptr<nvinfer1::ICudaEngine> engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());

    // -----------------------
    // Create buffer manager
    // -----------------------
    mBufManagers.push_back(std::make_unique<BufferManager>(engine));

    // ---------------
    // Create context
    // ---------------
    mContexts.push_back(
        UniquePtrTRT<nvinfer1::IExecutionContext>(engine->createExecutionContext()));
}

std::vector<std::string> TaillightInferenceAgent::infer(std::vector<std::string> imgPaths) {

    std::vector<std::string> resultStates;

    // ------------
    // Load Images
    // ------------
    int seqLen = imgPaths.size();

    std::vector<cv::Mat> imgSeq;

    // head padding
    for (int i = 0; i < 8; ++i) {
        imgSeq.push_back(cv::Mat(112, 112, CV_32FC3, 0.0));
    }

    // load images
    for (const auto &elem : imgPaths) {
        cv::Mat img = cv::imread(elem);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(112, 112));
        imgSeq.push_back(img);
    }

    // tail padding
    for (int i = 0; i < 7; ++i) {
        imgSeq.push_back(cv::Mat(112, 112, CV_32FC3, 0.0));
    }

    if (seqLen + 15 != static_cast<int>(imgSeq.size())) {
        std::cout << "오류" << std::endl;
    }

    // normalize
    for (auto &elem : imgSeq) {
        elem = (elem - 0.5) * 4.0;
    }

    std::vector<float> hostInBuffer(1 * 16 * 3 * 112 * 112);
    std::vector<float> hostMidBuffer(8 * 16 * 64 * 28 * 28);
    for (int iSeq = 0; iSeq < seqLen; ++iSeq) {
        // -------------------
        // Prepare Input Data
        // -------------------
        int idx = 0;
        for (int t = iSeq; t < iSeq + 16; ++t) {
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 112; ++h) {
                    for (int w = 0; w < 112; ++w) {
                        hostInBuffer[idx] = imgSeq[t].at<cv::Vec3f>(h, w)[c];
                        idx += 1;
                    }
                }
            }
        }
        // std::fill(hostInBuffer.begin(), hostInBuffer.end(), 1.0); // dummy test

        // ----------------------
        // Copy (Host -> Device)
        // ----------------------
        mBufManagers[0]->memcpy(true, mParams.inputTensorName, hostInBuffer.data());

        // -------------
        // Execute Unet
        // -------------
        std::vector<void *> buffers1 = mBufManagers[0]->getDeviceBindings();
        mContexts[0]->executeV2(buffers1.data());

        // --------------------------------
        // Copy (Device -> Host -> Device)
        // --------------------------------
        mBufManagers[0]->memcpy(false, mParams.outputTensorName, hostMidBuffer.data());
        mBufManagers[1]->memcpy(true, mParams.inputTensorName, hostMidBuffer.data());

        // ---------------
        // Execute 3Dconv
        // ---------------
        std::vector<void *> buffers2 = mBufManagers[1]->getDeviceBindings();
        mContexts[1]->executeV2(buffers2.data());

        // ----------------------
        // Copy (Device -> Host)
        // ----------------------
        std::vector<int> hostOutBuffer(8);
        mBufManagers[1]->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

        resultStates.push_back(mStates[hostOutBuffer[0]]);
    }
    return resultStates;
}

void TaillightInferenceAgent::benchmark_speed() {
    std::vector<float> hostInBuffer(1 * 16 * 3 * 112 * 112);
    std::vector<float> hostMidBuffer(8 * 16 * 64 * 28 * 28);

    for (int i = 0; i < 1000; ++i) {

        std::fill(hostInBuffer.begin(), hostInBuffer.end(), i); // dummy test

        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

        // ----------------------
        // Copy (Host -> Device)
        // ----------------------
        mBufManagers[0]->memcpy(true, mParams.inputTensorName, hostInBuffer.data());

        // -------------
        // Execute Unet
        // -------------
        std::vector<void *> buffers1 = mBufManagers[0]->getDeviceBindings();
        mContexts[0]->executeV2(buffers1.data());

        // --------------------------------
        // Copy (Device -> Host -> Device)
        // --------------------------------
        mBufManagers[0]->memcpy(false, mParams.outputTensorName, hostMidBuffer.data());
        mBufManagers[1]->memcpy(true, mParams.inputTensorName, hostMidBuffer.data());

        // ---------------
        // Execute 3Dconv
        // ---------------
        std::vector<void *> buffers2 = mBufManagers[1]->getDeviceBindings();
        mContexts[1]->executeV2(buffers2.data());

        // ----------------------
        // Copy (Device -> Host)
        // ----------------------
        std::vector<int> hostOutBuffer(8);
        mBufManagers[1]->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        std::cout << "processing_time (micro sec): " << duration << std::endl;
    }
}

int main() {
    SampleParams params;
    params.onnxFilePath.push_back(
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/Output/taillight_unet.onnx");
    params.onnxFilePath.push_back(
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/Output/taillight_3Dconv.onnx");
    params.inputFileDir =
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/Data/ETRI_CROP_TIGHT";
    params.inputTensorName = "Input";
    params.outputTensorName = "Output";

    TaillightInferenceAgent agent(params);
    agent.build(params.onnxFilePath[0]);
    agent.build(params.onnxFilePath[1]);

    // ----------
    // Infer Seq
    // ----------
    fs::remove_all("Output");
    for (const auto &entryFolder : fs::directory_iterator(params.inputFileDir)) {
        // Collect img paths of a sequence
        std::vector<std::string> imgPaths;
        for (const auto &entryFile : fs::directory_iterator(entryFolder.path())) {
            imgPaths.push_back(entryFile.path());
        }
        std::sort(imgPaths.begin(), imgPaths.end());

        // Infer
        std::vector<std::string> resultStates = agent.infer(imgPaths);

        // Save results
        fs::path targetParent = fs::path{"Output"} / entryFolder.path().filename();
        fs::create_directories(targetParent);
        for (int i = 0; i < static_cast<int>(imgPaths.size()); ++i) {
            fs::path targetPath = targetParent / (fs::path{imgPaths[i]}.stem().string() + "_" +
                                                  resultStates[i] + ".png");
            fs::copy_file(imgPaths[i], targetPath, fs::copy_options::overwrite_existing);
        }
        std::cout << targetParent << " Completed " << std::endl;
    }

    // ----------------
    // Benchmark speed
    // -----------------
    agent.benchmark_speed();

    return 0;
}
