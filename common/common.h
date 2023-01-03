#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H
#include <memory>
#include <numeric>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
struct InferDeleter {
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

namespace samplesCommon {

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    return 0;
}

inline int64_t volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
} // namespace samplesCommon
#endif // TENSORRT_COMMON_H