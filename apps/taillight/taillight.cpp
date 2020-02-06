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
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{false}; //!< Allow running the network in FP16 mode.
    std::string inputTensorName;
    std::string outputTensorName;
    std::string onnxFilePath;
    std::string inputFileDir;
};

class TaillightInferenceAgent {

  public:
    TaillightInferenceAgent(const SampleParams &params) : mParams(params) {}

    void build();
    std::vector<std::string> infer(std::vector<std::string> imgPaths);

  private:
    SampleParams mParams;
    const std::vector<std::string> mStates{"OOO", "BOO", "OLO", "BLO", "OOR", "BOR", "OLR", "BLR"};

    std::unique_ptr<BufferManager> mBufManager{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtrTRT<nvinfer1::IExecutionContext> mContext{nullptr};
};

void TaillightInferenceAgent::build() {
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
    parser->parseFromFile(mParams.onnxFilePath.c_str(),
                          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = UniquePtrTRT<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30);
    /*
    builder->setMaxBatchSize(mParams.batchSize);
    if (mParams.fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }
    if (mParams.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    */

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());

    // -----------------------
    // Create buffer manager
    // -----------------------
    mBufManager = std::make_unique<BufferManager>(mEngine);

    // ---------------
    // Create context
    // ---------------
    mContext = UniquePtrTRT<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
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

    std::vector<float> hostInBuffer(8 * 16 * 3 * 112 * 112);
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
        mBufManager->memcpy(true, mParams.inputTensorName, hostInBuffer.data());

        // --------
        // Execute
        // --------
        std::vector<void *> buffers = mBufManager->getDeviceBindings();
        mContext->executeV2(buffers.data());

        // ----------------------
        // Copy (Device -> Host)
        // ----------------------
        std::vector<float> hostOutBuffer(8 * 8);
        mBufManager->memcpy(false, mParams.outputTensorName, hostOutBuffer.data());

        // -------
        // Argmax
        // -------
        int argmaxIdx = 0;
        float maxVal = hostOutBuffer[0];
        for (int i = 1; i < 8; ++i) {
            if (hostOutBuffer[i] > maxVal) {
                argmaxIdx = i;
                maxVal = hostOutBuffer[i];
            }
        }
        resultStates.push_back(mStates[argmaxIdx]);
    }
    return resultStates;
}

int main() {
    SampleParams params;
    params.onnxFilePath =
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/Output/taillight.onnx";
    params.inputFileDir =
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/Data/ETRI_CROP_TIGHT";
    params.inputTensorName = "Input";
    params.outputTensorName = "Output";

    TaillightInferenceAgent agent(params);
    agent.build();

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

    return 0;

    /*
    for (int i = 0; i < 1000; ++i) {
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        // agent.infer();
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        std::cout << "processing_time (micro sec): " << duration << std::endl;
    }
    */
}
