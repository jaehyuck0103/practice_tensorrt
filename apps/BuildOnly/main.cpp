#include "TensorRT-OSS/samples/common/sampleEngines.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void build(const std::string &onnxFilePath) {
    sample::ModelOptions modelOption;
    modelOption.baseModel.model = onnxFilePath;
    modelOption.baseModel.format = sample::ModelFormat::kONNX;

    sample::BuildOptions buildOption;
    buildOption.workspace = 10 * 1024;
    buildOption.tf32 = false;
    buildOption.fp16 = true;
    buildOption.int8 = false;
    buildOption.save = true;
    buildOption.engine = fs::path{onnxFilePath}.replace_extension(".trt").string();

    sample::SystemOptions sysOption;

    // Build and Save Engine
    sample::BuildEnvironment env;
    getEngineBuildEnv(modelOption, buildOption, sysOption, env, std::cout);

    // network released after parser! parser destructor depends on network.
    env.parser = {};
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
        build(elem);
    }
}
