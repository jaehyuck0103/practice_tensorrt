#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <cassert>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>


using namespace std;

struct SampleParams
{
    int batchSize{1};                  //!< Number of inputs in a batch
    int dlaCore{-1};                   //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFilePath;
    std::string inputFilePath;
};

class Logger: public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std:: endl;
    }
} gLogger;

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

public:
    SampleOnnxMNIST(const SampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    void build();
    void infer();

private:
    SampleParams mParams;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    SampleUniquePtr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

void SampleOnnxMNIST::build()
{
    // ----------------------------
    // Create builder and network
    // ----------------------------
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // -------------------
    // Create ONNX parser
    // -------------------
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(mParams.onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // -------------
    // Build engine
    // -------------
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	builder->setMaxBatchSize(mParams.batchSize);
    /*
    config->setMaxWorkspaceSize(1 << 30);
    if (mParams.fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }
    if (mParams.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    */

    mEngine = SampleUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    // -----------
    // Validate
    // -----------
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    std::cout << network->getInput(0)->getName() << std::endl;
    std::cout << network->getOutput(0)->getName() << std::endl;
}

void SampleOnnxMNIST::infer()
{
    // --------------
    // Prepare Data
    // --------------
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(mParams.inputFilePath, fileData.data(), inputH, inputW);

    std::vector<float> hostInBuffer(inputH * inputW);
    for (int i = 0; i < inputH * inputW; ++i)
    {
        hostInBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    void* deviceInBuffer{nullptr};
    cudaMalloc(&deviceInBuffer, inputH * inputW * 4);
    
    void* deviceOutBuffer{nullptr};
    cudaMalloc(&deviceOutBuffer, 10 * 4);

    // ----------------------
    // Copy (Host -> Device)
    // ----------------------
    cudaMemcpy(deviceInBuffer, hostInBuffer.data(), inputH * inputW * 4, cudaMemcpyHostToDevice);

    // --------
    // Execute
    // --------
    int inputIndex = mEngine->getBindingIndex(mParams.inputTensorNames[0].c_str());
    int outputIndex = mEngine->getBindingIndex(mParams.outputTensorNames[0].c_str());
    void* buffers[2];
    buffers[inputIndex] = deviceInBuffer;
    buffers[outputIndex] = deviceOutBuffer;

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    context->executeV2(buffers);

    // ----------------------
    // Copy (Device -> Host)
    // ----------------------
    std::vector<float> hostOutBuffer(10);
    cudaMemcpy(hostOutBuffer.data(), deviceOutBuffer, 10 * 4, cudaMemcpyDeviceToHost);

    cout << "Result" << endl;
    for (const auto& elem: hostOutBuffer) { cout << elem << endl; }

    // -----
    // Free
    // -----
    cudaFree(deviceInBuffer);
    cudaFree(deviceOutBuffer);
}

int main()
{
    SampleParams params;
    params.onnxFilePath = "./data/mnist.onnx";
    params.inputFilePath = "./data/8.pgm";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");

    SampleOnnxMNIST sample(params);
    sample.build();
    sample.infer();
}
