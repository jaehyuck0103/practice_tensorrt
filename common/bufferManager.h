#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "./common.h"

class DeviceBuffer {
  public:
    DeviceBuffer(int numel, nvinfer1::DataType type) : mNumEl(numel), mType(type) {
        if (!allocFn(&mBuffer, this->nbBytes())) {
            throw std::bad_alloc();
        }
    }

    void *data() { return mBuffer; }

    const void *data() const { return mBuffer; }

    int nbBytes() const { return mNumEl * getTypeSize(mType); }

    ~DeviceBuffer() { freeFn(mBuffer); }

  private:
    int mNumEl{0};
    nvinfer1::DataType mType;
    void *mBuffer;

    bool allocFn(void **ptr, int byteSize) const {
        return cudaMalloc(ptr, byteSize) == cudaSuccess;
    }

    void freeFn(void *ptr) const { cudaFree(ptr); }
};

class BufferManager {
  public:
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine) : mEngine(engine) {
        // Create host and device buffers
        for (int i = 0; i < mEngine->getNbBindings(); i++) {
            auto dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            int vol = volume(dims);

            mDeviceBuffers.push_back(std::make_unique<DeviceBuffer>(vol, type));
        }
    }

    const std::vector<void *> getDeviceBindings() const {
        std::vector<void *> deviceBindings;
        for (const auto &elem : mDeviceBuffers) {
            deviceBindings.push_back(elem->data());
        }

        return deviceBindings;
    }

    void memcpy(const bool hostToDevice, const std::string &tensorName, void *hostPtr) {
        int index = getBindingIndex(tensorName);
        if (hostToDevice != mEngine->bindingIsInput(index)) {
            std::cout << "Memcpy: Wrong Direction." << std::endl;
            exit(1);
        }

        void *devicePtr = mDeviceBuffers[index]->data();
        const int byteSize = mDeviceBuffers[index]->nbBytes();

        void *dstPtr = hostToDevice ? devicePtr : hostPtr;
        const void *srcPtr = hostToDevice ? hostPtr : devicePtr;
        const cudaMemcpyKind memcpyType =
            hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;

        if (cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType) != cudaSuccess) {
            std::cout << "cudaMemcpy Failed" << std::endl;
            exit(1);
        }
    }

    ~BufferManager() = default;

  private:
    int getBindingIndex(const std::string &tensorName) const {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1) {
            std::cout << "Wrong Tensor Name" << std::endl;
            exit(1);
        }
        return index;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The pointer to the engine
    std::vector<std::unique_ptr<DeviceBuffer>> mDeviceBuffers;
};
