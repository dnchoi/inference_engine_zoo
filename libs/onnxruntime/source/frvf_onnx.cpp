#include <frvf_onnx.h>

using namespace onnx_frvf;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


inline std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

/**
 * @brief Construct a new frvf onnx::frvf onnx object
 * 
 * @param file_path 
 * @param useCUDA 
 * @param OPT_OPTION 
 */
frvf_onnx::frvf_onnx(const char * file_path, bool useCUDA, 
                     int OPT_OPTION, int B, int C, int W, int H){
    SPDLOG_INFO(file_path);
    SPDLOG_INFO(useCUDA);
    SPDLOG_INFO(OPT_OPTION);
    SPDLOG_INFO(B);
    SPDLOG_INFO(C);
    SPDLOG_INFO(W);
    SPDLOG_INFO(H);
    
    this->_Instance(file_path, useCUDA, OPT_OPTION, B, C, W, H);
}

inline GraphOptimizationLevel frvf_onnx::optimizer_selector(int expression){
    GraphOptimizationLevel a;
    switch (expression)
    {
    case 0:
        a = GraphOptimizationLevel::ORT_DISABLE_ALL;
        break;
    case 1:
        a = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        break;
    case 2:
        a = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        break;
    case 3:
        a = GraphOptimizationLevel::ORT_ENABLE_ALL;
        break;
    default:
        a = GraphOptimizationLevel::ORT_DISABLE_ALL;
        break;
    }

    return a;
}

void frvf_onnx::_Instance(const char * file_path, bool useCUDA, 
                          int OPT_OPTION, int B, int C, int W, int H){
    std::string modelFilepath(file_path);
    SPDLOG_INFO(modelFilepath);
    SPDLOG_INFO(useCUDA);
    SPDLOG_INFO(OPT_OPTION);
    std::string instanceName{"ONNX-face-recognition"};
    sessionOptions = new Ort::SessionOptions;
    sessionOptions->SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        // OrtCUDAProviderOptions cuda_options{0};
        // sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
        SPDLOG_INFO("Mac OS is not used CUDA");
    }
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    sessionOptions->SetGraphOptimizationLevel(optimizer_selector(OPT_OPTION));
    sess = new Ort::Session(*env, modelFilepath.c_str(), *sessionOptions);
    allocator = new Ort::AllocatorWithDefaultOptions;
    numInputNodes = sess->GetInputCount();
    numOutputNodes = sess->GetOutputCount();
    inputName = sess->GetInputName(0, *allocator);
    Ort::TypeInfo inputTypeInfo = sess->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    inputDims = inputTensorInfo.GetShape();
    outputName = sess->GetOutputName(0, *allocator);
    Ort::TypeInfo outputTypeInfo = sess->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    outputDims = outputTensorInfo.GetShape();
    
    if(inputDims[0] == -1){
        inputDims[0] = B;
    }
    if(outputDims[0] == -1){
        outputDims[0] = B;
    }
    SPDLOG_INFO("Number of Input Nodes : {}",numInputNodes);
    SPDLOG_INFO("Number of Output Nodes : {}",numOutputNodes);
    SPDLOG_INFO("Input Name : {}",inputName);
    SPDLOG_INFO("Input Type : {}",inputType);
    SPDLOG_INFO("Input Dimensions : {} {} {} {}",inputDims[0],inputDims[1],inputDims[2],inputDims[3]);
    SPDLOG_INFO("Output Name : {}",outputName);
    SPDLOG_INFO("Output Type : {}",outputType);
    SPDLOG_INFO("Output Dimensions : {} {} {} {}",outputDims[0],outputDims[1],outputDims[2],outputDims[3]);

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);
    SPDLOG_INFO("Instance Done");

    // cv::Mat channels[3];
    // cv::split(resizedImage, channels);
    // // Normalization per channel
    // // Normalization parameters obtained from
    // // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    // channels[0] = (channels[0] - 0.485) / 0.229;
    // channels[1] = (channels[1] - 0.456) / 0.224;
    // channels[2] = (channels[2] - 0.406) / 0.225;
    // cv::merge(channels, 3, resizedImage);
    // HWC to CHW
}

inline cv::Mat frvf_onnx::pre_processing(cv::Mat frame){
    cv::Mat preprocessedImage;
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage; 

    cv::resize(frame, resizedImageBGR,
            cv::Size(inputDims.at(2), inputDims.at(3)),
            cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    assert((int64_t)W == inputDims.at(2));
    assert((int64_t)H == inputDims.at(3));
    return preprocessedImage;
}

float frvf_onnx::do_inference(cv::Mat frame){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cv::Mat image = pre_processing(frame);
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(image.begin<float>(),
                             image.end<float>());
                             
    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));


    sess->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    // for (int _i=0; _i < outputDims[1]; _i ++)
    // {
    //     SPDLOG_INFO("{}",outputTensorValues.at(_i));
    // }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float processtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    return processtime;
}

frvf_onnx::~frvf_onnx(){
    
}
