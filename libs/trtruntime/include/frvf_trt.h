#ifndef __FRVF_TRT_H__
#define __FRVF_TRT_H__

#include <iostream>
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
#include <trt_utils.h>
#include <trt_logger.h>

namespace trt_frvf{

    class frvf_trt
    {
    private:
        // Ort::Env *env;
        // Ort::Session *sess;
        // Ort::SessionOptions *sessionOptions;
        // Ort::AllocatorWithDefaultOptions *allocator;
        // GraphOptimizationLevel optimizer_selector(int expression);
        
        std::vector<int64_t> inputDims;
        size_t numInputNodes;
        const char* inputName;

        std::vector<int64_t> outputDims;
        size_t numOutputNodes;
        const char* outputName;
        
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;
        
        nvinfer1::Dims *nodeDims;
        int total_node_number;
        float *data;
        float *prob; 

        void _Instance(const char * file_path, bool useCUDA, int OPT_OPTION, int B, int C, int W, int H);
        // void parseOnnxModel(const std::string& model_path, bool useCUDA, int OPT_OPTION, int B, int C, int W, int H);
        // std::string createEngine(const std::string& model_path, bool useCUDA, 
						//  int OPT_OPTION, int B, int C, int W, int H, std::string outname);
        nvinfer1::ICudaEngine* createEngine();
        nvinfer1::IHostMemory* createModelStream(nvinfer1::ICudaEngine *_engine);
        void loadModel();
        void parseOnnxModel();
        nvinfer1::IRuntime* runtime;
        cudaStream_t stream;
        nvinfer1::ICudaEngine * engine;
        nvinfer1::IExecutionContext * context;
        // nvinfer1::ICudaEngine * engine;
        // nvinfer1::IExecutionContext * context;
        // TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
        // TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
        int input_B,input_C,input_W,input_H;
        std::string modelName, trtName;

    public:
        frvf_trt(const char * file_path, bool useCUDA, int OPT_OPTION, int B, int C, int W, int H);
        ~frvf_trt();

        float do_inference(cv::Mat frame);
    };
}

#endif // __FRVF_TRT_H__