#include <experimental/filesystem> // gcc 8부터 experimental 뗄 수 있다. CMAKE의 link도 나중에 같이 떼주도록 하자.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "common/common.h"

namespace fs = std::experimental::filesystem;

struct SampleParams {
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{false}; //!< Allow running the network in FP16 mode.
    std::string onnxFilePath;
    std::string engineFilePath;
};

void build(const SampleParams &params) {
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
    parser->parseFromFile(params.onnxFilePath.c_str(),
                          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = UniquePtrTRT<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30);
    /*
    builder->setMaxBatchSize(params.batchSize);
    if (params.fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }
    if (params.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), params.dlaCore);
    */

    std::shared_ptr<nvinfer1::ICudaEngine> engine{nullptr};
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());

    // -----------------
    // Serialize engine
    // -----------------
    UniquePtrTRT<nvinfer1::IHostMemory> serializedEngine{engine->serialize()};

    std::string engineFilePath = fs::path{params.onnxFilePath}.replace_extension(".trt");
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());
}

int main() {
    // Get Onnx lists
    std::ifstream f("./InputOnnxList.txt");
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(f, line)) {
        lines.push_back(line);
    }

    // Build each onnx
    for (const auto &elem : lines) {
        SampleParams params;
        params.onnxFilePath = elem;

        build(params);
    }
}
