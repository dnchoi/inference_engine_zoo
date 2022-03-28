#ifndef __FRVF_ONNX_H__
#define __FRVF_ONNX_H__

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace onnx_frvf{

    class frvf_onnx
    {
    private:
        Ort::Env *env;
        Ort::Session *sess;
        Ort::SessionOptions *sessionOptions;
        Ort::AllocatorWithDefaultOptions *allocator;
        inline GraphOptimizationLevel optimizer_selector(int expression);
        
        std::vector<int64_t> inputDims;
        size_t numInputNodes;
        const char* inputName;

        std::vector<int64_t> outputDims;
        size_t numOutputNodes;
        const char* outputName;
        
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;
        inline cv::Mat pre_processing(cv::Mat frame);
        void _Instance(const char * file_path, bool useCUDA, int OPT_OPTION, int B, int C, int W, int H);
    public:
        frvf_onnx(const char * file_path, bool useCUDA, int OPT_OPTION, int B, int C, int W, int H);
        ~frvf_onnx();
        float do_inference(cv::Mat frame);
    };
}

#endif // __FRVF_ONNX_H__