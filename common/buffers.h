#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H
#include "NvInfer.h"
#include "common/common.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
class DeviceAllocator {
  public:
    bool operator()(void **ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};
class DeviceFree {
  public:
    void operator()(void *ptr) const { cudaFree(ptr); }
};

class HostAllocator {
  public:
    bool operator()(void **ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
  public:
    void operator()(void *ptr) const { free(ptr); }
};

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
  public:
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize_(0), mCapacity_(0), mType_(type), mBuffer_(nullptr)
    {
    }

    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize_(size), mCapacity_(size), mType_(type)
    {
        if (!allocFn_(&mBuffer_, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    // 移动构造
    GenericBuffer(GenericBuffer &&buf)
        : mSize_(buf.mSize_), mCapacity_(buf.mCapacity_), mType_(buf.mType_),
          mBuffer_(buf.mBuffer_)
    {
        buf.mSize_ = 0;
        buf.mCapacity_ = 0;
        buf.mType_ = nvinfer1::DataType::kFLOAT;
        buf.mBuffer_ = nullptr;
    }

    GenericBuffer &operator=(GenericBuffer &&buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer_);
            mSize_ = buf.mSize_;
            mCapacity_ = buf.mCapacity_;
            mType_ = buf.mType_;
            mBuffer_ = buf.mBuffer_;
            // Reset buf.
            buf.mSize_ = 0;
            buf.mCapacity_ = 0;
            buf.mBuffer_ = nullptr;
        }
        return *this;
    }

    void *data() { return mBuffer_; }
    const void *data() const { return mBuffer_; }
    size_t size() const { return mSize_; }
    size_t nbBytes() const
    {
        return this->size() * samplesCommon::getElementSize(mType_);
    }

    void resize(size_t newSize)
    {
        mSize_ = newSize;
        if (mCapacity_ < newSize)
        {
            freeFn(mBuffer_);
            if (!allocFn(&mBuffer_, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity_ = newSize;
        }
    }

    void resize(const nvinfer1::Dims &dims)
    {
        return this->resize(samplesCommon::volume(dims));
    }
    ~GenericBuffer() { freeFn(mBuffer_); }

  private:
    size_t mSize_{0}, mCapacity_{0};
    nvinfer1::DataType mType_;
    void *mBuffer_;
    AllocFunc allocFn_;
    FreeFunc freeFn_;
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer {
  public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

// The BufferManager class handles host and device buffer allocation and
// deellocation
class BufferManager {
  public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                  const int batchSize = 0,
                  const nvinfer1::IExecutionContext *context = nullptr)
        : mEngine_(engine), mBatchSize_(batchSize)
    {
        assert(engine->hasImplicitBatchDimension() || mBatchSize_ == 0);
        // create host and device buffers
        for (int i = 0; i < mEngine_->getNbBindings(); i++)
        {
            auto dims = context ? context->getBindingDimensions(i)
                                : mEngine_->getBindingDimensions(i);
            size_t vol =
                context || !mBatchSize_ ? 1 : static_cast<size_t>(mBatchSize_);
            nvinfer1::DataType type = mEngine_->getBindingDataType(i);
            int vecDim = mEngine_->getBindingVectorizedDim(i);
            if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            {
                int scalarsPerVec = mEngine_->getBindingComponentsPerElement(i);
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= samplesCommon::volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            mDeviceBindings_.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers_.emplace_back(std::move(manBuf));
        }
    }

  public:
    //!< The pointer to the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;
    //!< The batch size for legacy networks, 0 otherwise.
    int mBatchSize_;
    //!< The vector of pointers to managed buffers
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers_;
    //!< The vector of device buffers needed for engine execution
    std::vector<void *> mDeviceBindings_;
};

#endif