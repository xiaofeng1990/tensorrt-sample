#ifndef SAMPLE_ONNX_MNIST_H
#define SAMPLE_ONNX_MNIST_H
#include "NvInfer.h"
#include "common/argsParser.h"
#include "common/common.h"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

class SampleOnnxMNIST {
  public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams &params)
        : mParams_(params), mEngine_(nullptr)
    {
    }
    bool Build();
    bool Infer();

  private:
    samplesCommon::OnnxSampleParams mParams_;
    // the dimensions of the input to the network
    nvinfer1::Dims mInputDims_;
    // the dimensions of the output to the network
    nvinfer1::Dims mOutputDms_;
    // the number to classify
    int mNumber_{0};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;

    // Parses an ONNX model for MNIST and creates a TensorRT network
    bool ConstructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                          SampleUniquePtr<nvonnxparser::IParser> &parser);

    // Reads the input  and stores the result in a managed buffer
    bool ProcessInput(const samplesCommon::BufferManager &buffers);
    bool VerifyOutput(const samplesCommon::BufferManager &buffers);
};
#endif