#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <string>
#include <vector>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace samplesCommon {
struct SampleParams {
    int32_t batchSize{1};
    int32_t dlaCore{-1};
    bool int8{false};
    bool fp16{false};
    std::vector<std::string> dataDirs;
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

struct OnnxSampleParams : public SampleParams {
    std::string onnxFileName;
};

} // namespace samplesCommon
#endif