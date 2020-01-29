/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "NvInfer.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

inline int volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

inline int getTypeSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    cout << "Invalid DataType." << endl;
    exit(1);
}

class DeviceBuffer
{
public:
    DeviceBuffer(int size, nvinfer1::DataType type)
        : mSize(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    void* data()
    {
        return mBuffer;
    }

    const void* data() const
    {
        return mBuffer;
    }

    int size() const
    {
        return mSize;
    }

    int nbBytes() const
    {
        return this->size() * getTypeSize(mType);
    }

    ~DeviceBuffer()
    {
        freeFn(mBuffer);
    }

private:
    int mSize{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    
    bool allocFn(void** ptr, int size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }

    void freeFn(void* ptr) const
    {
        cudaFree(ptr);
    }
};


class BufferManager
{
public:
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine)
        : mEngine(engine)
    {
        // Create host and device buffers
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            auto dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            int vol = mEngine->getMaxBatchSize() * volume(dims);

            mDeviceBuffers.push_back(make_unique<DeviceBuffer>(vol, type));
            mDeviceBindings.push_back(mDeviceBuffers.back()->data());
        }
    }

    /*
    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }
    */

    const std::vector<void*>& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    void memcpy(const string& tensorName, const bool hostToDevice, void* hostPtr)
    {
        int index = getBindingIndex(tensorName);
        if (hostToDevice != mEngine->bindingIsInput(index)) {
            cout << "Memcpy: Wrong Direction." << endl;
            exit(1);
        }

        void* devicePtr = mDeviceBuffers[index]->data();
        const int byteSize = mDeviceBuffers[index]->nbBytes();

        void* dstPtr = hostToDevice ? devicePtr : hostPtr;
        const void* srcPtr = hostToDevice ? hostPtr : devicePtr;
        const cudaMemcpyKind memcpyType = hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;

        if (cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType) != cudaSuccess) {
            cout << "cudaMemcpy Failed" << endl;
            exit(1);
        }
    }

    ~BufferManager() = default;

private:
    int getBindingIndex(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1) {
            cout << "Wrong Tensor Name" << endl;
            exit(1);
        }
        return index;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    std::vector<std::unique_ptr<DeviceBuffer>> mDeviceBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mDeviceBindings; //!< The vector of device buffers needed for engine execution
};
