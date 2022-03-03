#include <frvf_onnx.h>

using namespace onnx_frvf;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
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


std::ostream& operator<<(std::ostream& os,
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
frvf_onnx::frvf_onnx(std::string file_path, bool useCUDA, int OPT_OPTION){
    
    this->_Instance(file_path, useCUDA, OPT_OPTION);
}

GraphOptimizationLevel frvf_onnx::optimizer_selector(int expression){
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

void frvf_onnx::_Instance(std::string file_path, bool useCUDA, int OPT_OPTION)
{
#ifdef _DEBUG_
    std::string modelFilepath = file_path;
    std::string instanceName{"ONNX-face-recognition"};
    sessionOptions = new Ort::SessionOptions;
    sessionOptions->SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        OrtCUDAProviderOptions cuda_options{0};
        sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
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

#else
    std::string modelFilepath = file_path;
    std::string instanceName{"ONNX-face-recognition"};
    Ort::SessionOptions sessionOptions;

    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        OrtCUDAProviderOptions cuda_options{0};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;


    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;
#endif
}

float frvf_onnx::do_inference(std::string imageFilepath){
    cv::Mat imageBGR2= cv::Mat::zeros(1, 1, CV_64F);
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

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
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());
                             
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
   
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sess->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    for (int _i=0; _i < outputDims[1]; _i ++)
    {
        SPDLOG_TRACE("{}",outputTensorValues.at(_i));
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float processtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    return processtime;
}

frvf_onnx::~frvf_onnx(){
    
}