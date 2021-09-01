#include "TensorRT-OSS/samples/common/buffers.h"
#include "TensorRT-OSS/samples/common/common.h"
#include "TensorRT-OSS/samples/common/sampleEngines.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct SampleParams {
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{true};  //!< Allow running the network in FP16 mode.
    std::string onnxFilePath;
    std::string engineFilePath;
};

bool build(const SampleParams &params) {
    // ----------------------------
    // Create builder and network
    // ----------------------------
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

    if (!builder) {
        return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // ------------------------
    // Create builder config
    // ------------------------
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    config->setMaxWorkspaceSize(1 << 30);
    if (params.fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto parsed = parser->parseFromFile(
        params.onnxFilePath.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        return false;
    }

    // -------------
    // Build engine
    // -------------
    std::unique_ptr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> engine{nullptr};
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) {
        return false;
    }

    // -----------------
    // Save engine
    // -----------------
    const std::string engineFilePath =
        fs::path{params.onnxFilePath}.replace_extension(".trt").string();
    return sample::saveEngine(*engine, engineFilePath, std::cout);
}

int main() {

    // Get Onnx lists
    const std::string homeDir = std::getenv("HOME");
    std::ifstream f("./InputOnnxList.txt");
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(f, line)) {
        if (line.rfind("~", 0) == 0) {
            line = homeDir + line.substr(1);
        }

        lines.push_back(line);
        std::cout << line << std::endl;
    }

    // Build each onnx
    for (const auto &elem : lines) {
        SampleParams params;
        params.onnxFilePath = elem;

        build(params);
    }
}
