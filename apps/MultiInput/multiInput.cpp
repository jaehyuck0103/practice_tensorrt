#include "TensorRT-OSS/samples/common/common.h"
#include "TensorRT-OSS/samples/common/sampleEngines.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

class SampleMultiInput {
  public:
    bool build();
    void infer();

  private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
};

bool SampleMultiInput::build() {

    // ---------------
    // Build Engine
    // ---------------
    {
        sample::ModelOptions modelOption;
        modelOption.baseModel.model = "./multi_input.onnx";
        modelOption.baseModel.format = sample::ModelFormat::kONNX;

        sample::BuildOptions buildOption;
        sample::SystemOptions sysOption;

        // Get Engine
        sample::BuildEnvironment env;
        getEngineBuildEnv(modelOption, buildOption, sysOption, env, std::cout);

        mEngine = std::move(env.engine);

        // network released after parser! parser destructor depends on network.
        env.parser = {};
    }

    // ---------------
    // Create context
    // ---------------
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    return true;
}

void SampleMultiInput::infer() {
    // -------------------
    // Check Tensornames
    // -------------------
    std::cout << "\n\nCheck TensorNames\n";
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
        std::string tensorType = mEngine->bindingIsInput(i) ? "input: " : "output: ";
        std::cout << tensorType << mEngine->getBindingName(i) << "\n";
    }
    std::cout << "\n\n";

    // -----------------------
    // Prepare Device Buffers
    // -----------------------
    int byteSize = 9 * 4;
    void *deviceBuffer_x1;
    void *deviceBuffer_x2;
    void *deviceBuffer_y;
    cudaMalloc(&deviceBuffer_x1, byteSize);
    cudaMalloc(&deviceBuffer_x2, byteSize);
    cudaMalloc(&deviceBuffer_y, byteSize);

    // ---------------------
    // Copy (Host -> Device)
    // ---------------------
    std::vector<float> x1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> x2{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    cudaMemcpy(deviceBuffer_x1, x1.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBuffer_x2, x2.data(), byteSize, cudaMemcpyHostToDevice);

    // --------
    // Execute
    // --------
    std::vector<void *> buffers{deviceBuffer_x1, deviceBuffer_x2, deviceBuffer_y};
    mContext->executeV2(buffers.data());

    // ----------------------
    // Copy (Device -> Host)
    // ----------------------
    std::vector<float> hostOutBuffer(9);
    cudaMemcpy(hostOutBuffer.data(), deviceBuffer_y, byteSize, cudaMemcpyDeviceToHost);

    // -------------
    // Print Result
    // -------------
    std::cout << "Input1\n";
    for (const auto &elem : x1) {
        std::cout << elem << "\n";
    }
    std::cout << "Input2\n";
    for (const auto &elem : x2) {
        std::cout << elem << "\n";
    }
    std::cout << "Result = Input1 + Input2\n";
    for (const auto &elem : hostOutBuffer) {
        std::cout << elem << "\n";
    }

    // --------
    // Release
    // --------
    cudaFree(deviceBuffer_x1);
    cudaFree(deviceBuffer_x2);
    cudaFree(deviceBuffer_y);
}

int main() {
    SampleMultiInput sample;
    sample.build();
    sample.infer();
}
